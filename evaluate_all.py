"""
evaluate_all.py

Master evaluation script. Loads all trained checkpoints from Phases 2-4,
generates samples, computes every metric, saves structured results.

Usage:
    python evaluate_all.py --outputs_root outputs --results_dir results

Expects checkpoints at:
    outputs/{exp_name}/checkpoints/checkpoint_latest.pt
    outputs/{exp_name}/config.yaml
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import OmegaConf
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import (
    compute_ksd_squared, median_bandwidth,
    compute_stein_kernel_gradient, compute_stein_kernel_gradient_efficient,
)
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Metric computation
# ====================================================================

@torch.no_grad()
def compute_all_metrics(
    samples: torch.Tensor,
    energy,
    ref_samples: torch.Tensor = None,
    n_particles: int = None,
    spatial_dim: int = None,
    is_gmm: bool = False,
) -> dict:
    """Compute every metric for a set of terminal samples."""
    metrics = {}
    device = samples.device
    N, D = samples.shape

    # --- Energy statistics ---
    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

    # --- KSD ---
    # Need scores: compute via autograd
    samples_req = samples.detach().requires_grad_(True)
    E = energy.eval(samples_req)
    scores = -torch.autograd.grad(E.sum(), samples_req)[0]  # score = -∇E
    samples_req = samples_req.detach()

    ell = median_bandwidth(samples)
    N_ksd = min(N, 2000)
    idx = torch.randperm(N, device=device)[:N_ksd]
    metrics['ksd_squared'] = compute_ksd_squared(
        samples[idx], scores[idx].detach(), ell
    ).item()
    metrics['bandwidth'] = ell.item()

    # --- Reference-based metrics ---
    if ref_samples is not None:
        ref_samples = ref_samples.to(device)
        B = min(N, len(ref_samples))
        idx_g = torch.randperm(N, device=device)[:B]
        idx_r = torch.randperm(len(ref_samples), device=device)[:B]
        gen = samples[idx_g]
        ref = ref_samples[idx_r]

        # Energy W2
        ref_E = energy.eval(ref)
        metrics['ref_mean_energy'] = ref_E.mean().item()
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E.cpu().numpy(), gen_E[idx_g].cpu().numpy()) ** 0.5
        )

        # Particle-system metrics
        if n_particles is not None and n_particles > 1:
            gen_dist = interatomic_dist(gen, n_particles, spatial_dim)
            ref_dist = interatomic_dist(ref, n_particles, spatial_dim)
            metrics['dist_w2'] = float(pot.emd2_1d(
                gen_dist.cpu().numpy().reshape(-1),
                ref_dist.cpu().numpy().reshape(-1),
            ))

            M = dist_point_clouds(
                gen.reshape(-1, n_particles, spatial_dim).cpu(),
                ref.reshape(-1, n_particles, spatial_dim).cpu(),
            )
            a = torch.ones(M.shape[0]) / M.shape[0]
            b = torch.ones(M.shape[1]) / M.shape[1]
            metrics['eq_w2'] = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)

    # --- GMM-specific: mode coverage ---
    if is_gmm and hasattr(energy, 'count_modes_covered'):
        cov = energy.count_modes_covered(samples)
        metrics['n_modes_covered'] = cov['n_modes_covered']
        metrics['n_modes_total'] = cov['n_modes_total']
        metrics['coverage_fraction'] = cov['coverage_fraction']
        metrics['per_mode_counts'] = cov['per_mode_counts']

    # --- Store energy histogram data (as arrays for plotting later) ---
    metrics['_energy_values'] = gen_E.cpu().numpy().tolist()

    return metrics


def compute_chunking_timing(samples, scores, ell):
    """Compare wall-clock time of full vs chunked Stein kernel gradient."""
    timings = {}

    # Full
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    g_full = compute_stein_kernel_gradient(samples, scores, ell)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings['full_time'] = time.time() - t0

    # Chunked (chunk=128)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    g_chunk = compute_stein_kernel_gradient_efficient(samples, scores, ell, chunk_size=128)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings['chunk128_time'] = time.time() - t0

    # Chunked (chunk=256)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    g_chunk2 = compute_stein_kernel_gradient_efficient(samples, scores, ell, chunk_size=256)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings['chunk256_time'] = time.time() - t0

    # Verify equivalence
    timings['max_diff_chunk128'] = (g_full - g_chunk).abs().max().item()
    timings['max_diff_chunk256'] = (g_full - g_chunk2).abs().max().item()

    return timings


# ====================================================================
# Experiment loading and sample generation
# ====================================================================

def load_experiment(exp_name, outputs_root, device):
    """Load a trained experiment: config, model, energy, source."""
    exp_dir = Path(outputs_root) / exp_name
    cfg_path = exp_dir / 'config.yaml'
    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'

    if not cfg_path.exists():
        # Try Hydra's .hydra directory
        cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        print(f"  WARNING: No config found for {exp_name}")
        return None
    if not ckpt_path.exists():
        print(f"  WARNING: No checkpoint found for {exp_name}")
        return None

    cfg = OmegaConf.load(cfg_path)
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    # Build timestep config from the saved cfg
    ts_cfg = {}
    if hasattr(cfg, 'timesteps'):
        ts_cfg = {
            't0': cfg.timesteps.t0, 't1': cfg.timesteps.t1,
            'steps': cfg.timesteps.steps, 'rescale_t': cfg.timesteps.rescale_t,
        }
    else:
        # Reconstruct from top-level config
        ts_cfg = {
            't0': 0.0, 't1': 1.0,
            'steps': cfg.get('nfe', 200),
            'rescale_t': cfg.get('rescale_t', None),
        }

    return {
        'sde': sde, 'source': source, 'energy': energy,
        'ts_cfg': ts_cfg, 'cfg': cfg,
    }


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, batch_size, device):
    """Generate terminal samples."""
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1)
        n += b
    return torch.cat(x1_list)[:n_samples]


# ====================================================================
# Main evaluation loop
# ====================================================================

def evaluate_experiment_group(
    exp_names: list,
    outputs_root: str,
    device: str,
    n_samples: int = 2000,
    n_eval_seeds: int = 5,
) -> dict:
    """Evaluate a group of experiments (same benchmark, different methods/seeds)."""
    results = {}

    for exp_name in exp_names:
        print(f"\n  Evaluating {exp_name}...")
        exp = load_experiment(exp_name, outputs_root, device)
        if exp is None:
            continue

        cfg = exp['cfg']
        energy = exp['energy']
        is_gmm = 'rotgmm' in exp_name

        # Determine particle system properties
        n_particles = getattr(energy, 'n_particles', 1)
        spatial_dim = getattr(energy, 'n_spatial_dim', cfg.get('dim', None))

        # Load reference samples
        ref_samples = None
        if hasattr(cfg, 'evaluator') and hasattr(cfg.evaluator, 'ref_samples_path'):
            import os
            root = Path(os.path.abspath(__file__)).parent
            ref_path = root / cfg.evaluator.ref_samples_path
            if ref_path.exists():
                ref_np = np.load(ref_path, allow_pickle=True)
                ref_samples = torch.tensor(ref_np, dtype=torch.float32)
                if n_particles > 1:
                    ref_samples = remove_mean(ref_samples, n_particles, spatial_dim)
        elif hasattr(energy, 'get_ref_samples'):
            ref_samples = energy.get_ref_samples()

        # Multiple evaluation seeds
        seed_metrics = []
        batch_size = min(n_samples, cfg.get('eval_batch_size', 2000))

        for eval_seed in range(n_eval_seeds):
            torch.manual_seed(eval_seed * 7777)
            samples = generate_samples(
                exp['sde'], exp['source'], exp['ts_cfg'],
                n_samples, batch_size, device
            )
            m = compute_all_metrics(
                samples, energy, ref_samples,
                n_particles if n_particles > 1 else None,
                spatial_dim if n_particles > 1 else None,
                is_gmm=is_gmm,
            )
            seed_metrics.append(m)

        # Aggregate across eval seeds
        agg = {}
        for key in seed_metrics[0]:
            if key.startswith('_'):
                agg[key] = seed_metrics[0][key]  # Store first seed's raw data
                continue
            if isinstance(seed_metrics[0][key], (int, float)):
                vals = [m[key] for m in seed_metrics]
                agg[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
            elif isinstance(seed_metrics[0][key], list):
                agg[key] = seed_metrics[0][key]  # Lists: take first seed

        results[exp_name] = agg

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained experiments')
    parser.add_argument('--outputs_root', type=str, default='outputs')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_eval_seeds', type=int, default=5)
    parser.add_argument('--group', type=str, default=None,
                        help='Evaluate only this group (e.g., "dw4", "lj13")')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover all experiments
    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        print(f"ERROR: outputs root {outputs_root} does not exist")
        return

    all_experiments = sorted([
        d.name for d in outputs_root.iterdir()
        if d.is_dir() and (d / 'checkpoints').exists()
    ])
    print(f"Found {len(all_experiments)} experiments: {all_experiments}")

    # Group by benchmark
    groups = defaultdict(list)
    for exp in all_experiments:
        if exp.startswith('dw4'):
            groups['dw4'].append(exp)
        elif exp.startswith('lj13'):
            groups['lj13'].append(exp)
        elif exp.startswith('lj38'):
            groups['lj38'].append(exp)
        elif exp.startswith('lj55'):
            groups['lj55'].append(exp)
        elif exp.startswith('muller'):
            groups['muller'].append(exp)
        elif exp.startswith('blogreg'):
            groups['blogreg'].append(exp)
        elif exp.startswith('rotgmm'):
            dim = exp.split('_')[0]  # e.g., 'rotgmm10'
            groups[dim].append(exp)
        else:
            groups['other'].append(exp)

    # Filter if --group specified
    if args.group:
        groups = {k: v for k, v in groups.items() if k == args.group}

    # Evaluate each group
    all_results = {}
    for group_name, exp_names in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating group: {group_name} ({len(exp_names)} experiments)")
        print(f"{'='*60}")

        group_results = evaluate_experiment_group(
            exp_names, str(outputs_root), device,
            args.n_samples, args.n_eval_seeds,
        )
        all_results[group_name] = group_results

        # Save per-group results
        group_path = results_dir / f'{group_name}_results.json'
        with open(group_path, 'w') as f:
            json.dump(group_results, f, indent=2, default=str)
        print(f"  Saved to {group_path}")

    # --- Chunking timing test ---
    print(f"\n{'='*60}")
    print("Chunking timing test")
    print(f"{'='*60}")
    timing_results = {}
    for dim_label, dim in [('dw4_8d', 8), ('lj13_39d', 39), ('lj38_114d', 114), ('lj55_165d', 165)]:
        N = 512
        samples = torch.randn(N, dim, device=device)
        scores = -samples  # dummy scores
        ell = median_bandwidth(samples)
        t = compute_chunking_timing(samples, scores, ell)
        timing_results[dim_label] = t
        print(f"  {dim_label}: full={t['full_time']:.4f}s, "
              f"chunk128={t['chunk128_time']:.4f}s, "
              f"chunk256={t['chunk256_time']:.4f}s, "
              f"max_diff={t['max_diff_chunk128']:.2e}")

    with open(results_dir / 'chunking_timing.json', 'w') as f:
        json.dump(timing_results, f, indent=2)

    # Save master results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== All evaluation complete. Results in {results_dir}/ ===")


if __name__ == '__main__':
    main()
