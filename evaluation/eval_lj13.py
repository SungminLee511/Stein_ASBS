"""
eval_lj13.py — Evaluate LJ13 baseline and KSD-ASBS checkpoints.

Evaluates:
  1. lj13_asbs (baseline) — checkpoint_latest.pt
  2. lj13_ksd_asbs — checkpoint_50.pt  (early, before divergence)
  3. lj13_ksd_asbs — checkpoint_1550.pt (late, just before NaN)

Metrics: energy_W2, eq_W2, dist_W2, KSD², energy stats.
5 eval seeds × 2000 samples each.

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python evaluation/eval_lj13.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5


def load_model(config_path, ckpt_path, device):
    """Load model from config + checkpoint."""
    cfg = OmegaConf.load(config_path)
    # Resolve interpolations manually for the fields we need
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    energy = hydra.utils.instantiate(cfg_resolved.energy, device=device)
    source = hydra.utils.instantiate(cfg_resolved.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg_resolved.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg_resolved.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    ts_cfg = {
        't0': float(cfg_resolved.timesteps.t0),
        't1': float(cfg_resolved.timesteps.t1),
        'steps': int(cfg_resolved.nfe),
        'rescale_t': cfg_resolved.get('rescale_t', None),
    }

    return sde, source, energy, ts_cfg, cfg_resolved


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device, batch_size=500):
    """Generate terminal samples in batches."""
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1)
        n += b
        print(f"  Generated {n}/{n_samples} samples", end='\r')
    print()
    return torch.cat(x1_list)[:n_samples]


@torch.no_grad()
def compute_metrics(samples, energy, ref_samples, n_particles=13, spatial_dim=3):
    """Compute all LJ13 metrics. Handles NaN gracefully."""
    metrics = {}
    device = samples.device

    # Check for NaN/Inf
    nan_count = torch.isnan(samples).any(dim=1).sum().item()
    inf_count = torch.isinf(samples).any(dim=1).sum().item()
    metrics['nan_count'] = nan_count
    metrics['inf_count'] = inf_count

    # Filter valid samples
    valid_mask = ~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))
    valid_samples = samples[valid_mask]
    metrics['valid_count'] = valid_samples.shape[0]

    if valid_samples.shape[0] < 10:
        print(f"  WARNING: Only {valid_samples.shape[0]} valid samples!")
        return metrics

    # Energy stats
    gen_E = energy.eval(valid_samples)
    valid_E_mask = ~(torch.isnan(gen_E) | torch.isinf(gen_E))
    gen_E_valid = gen_E[valid_E_mask]

    metrics['mean_energy'] = gen_E_valid.mean().item()
    metrics['std_energy'] = gen_E_valid.std().item()
    metrics['min_energy'] = gen_E_valid.min().item()
    metrics['max_energy'] = gen_E_valid.max().item()

    # KSD
    try:
        with torch.enable_grad():
            s_req = valid_samples[:min(2000, len(valid_samples))].detach().requires_grad_(True)
            E_req = energy.eval(s_req)
            scores = -torch.autograd.grad(E_req.sum(), s_req)[0]
            s_req = s_req.detach()
        ell = median_bandwidth(s_req)
        metrics['ksd_squared'] = compute_ksd_squared(s_req, scores.detach(), ell).item()
        metrics['bandwidth'] = ell.item()
    except Exception as e:
        print(f"  KSD computation failed: {e}")
        metrics['ksd_squared'] = float('nan')

    # Reference-based metrics
    ref_samples = ref_samples.to(device)
    B = min(valid_samples.shape[0], len(ref_samples))
    idx_g = torch.randperm(valid_samples.shape[0], device=device)[:B]
    idx_r = torch.randperm(len(ref_samples), device=device)[:B]
    gen = valid_samples[idx_g]
    ref = ref_samples[idx_r]

    # Energy W2
    ref_E = energy.eval(ref)
    metrics['ref_mean_energy'] = ref_E.mean().item()
    try:
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E.cpu().numpy(), gen_E_valid[:B].cpu().numpy()) ** 0.5
        )
    except Exception as e:
        print(f"  energy_w2 failed: {e}")
        metrics['energy_w2'] = float('nan')

    # Interatomic distance W2
    try:
        gen_dist = interatomic_dist(gen, n_particles, spatial_dim)
        ref_dist = interatomic_dist(ref, n_particles, spatial_dim)
        metrics['dist_w2'] = float(pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        ))
    except Exception as e:
        print(f"  dist_w2 failed: {e}")
        metrics['dist_w2'] = float('nan')

    # Point cloud W2 (eq_w2) — subsample to avoid O(n²×k³) blowup
    EQ_W2_SUBSAMPLE = 200  # 200×200 distance matrix is manageable
    try:
        n_eq = min(EQ_W2_SUBSAMPLE, gen.shape[0], ref.shape[0])
        eq_gen = gen[:n_eq].reshape(-1, n_particles, spatial_dim).cpu()
        eq_ref = ref[:n_eq].reshape(-1, n_particles, spatial_dim).cpu()
        M = dist_point_clouds(eq_gen, eq_ref)
        if torch.isnan(M).any() or torch.isinf(M).any():
            print(f"  eq_w2: distance matrix has NaN/Inf, skipping")
            metrics['eq_w2'] = float('nan')
        else:
            a = torch.ones(M.shape[0]) / M.shape[0]
            b = torch.ones(M.shape[1]) / M.shape[1]
            metrics['eq_w2'] = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)
    except Exception as e:
        print(f"  eq_w2 failed: {e}")
        metrics['eq_w2'] = float('nan')

    return metrics


def evaluate_checkpoint(label, config_path, ckpt_path, ref_samples):
    """Evaluate a single checkpoint across multiple seeds."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    sde, source, energy, ts_cfg, cfg = load_model(config_path, ckpt_path, DEVICE)

    ref_torch = torch.tensor(ref_samples, dtype=torch.float32).to(DEVICE)
    ref_torch = remove_mean(ref_torch, 13, 3)

    seed_results = []
    for seed in range(N_EVAL_SEEDS):
        print(f"\n  Seed {seed}/{N_EVAL_SEEDS}:")
        torch.manual_seed(seed * 7777)
        np.random.seed(seed * 7777)

        samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE)
        samples = remove_mean(samples, 13, 3)
        m = compute_metrics(samples, energy, ref_torch)
        seed_results.append(m)

        # Print key metrics
        for k in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'mean_energy', 'valid_count']:
            if k in m:
                print(f"    {k}: {m[k]}")

    # Aggregate
    agg = {}
    numeric_keys = ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared',
                     'mean_energy', 'std_energy', 'min_energy', 'max_energy',
                     'ref_mean_energy', 'bandwidth', 'valid_count', 'nan_count', 'inf_count']
    for key in numeric_keys:
        vals = [m[key] for m in seed_results if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if vals:
            agg[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }

    return agg, seed_results


def main():
    print("LJ13 Evaluation Script")
    print(f"Device: {DEVICE}")
    print(f"N_SAMPLES: {N_SAMPLES}, N_EVAL_SEEDS: {N_EVAL_SEEDS}")

    # Load reference samples
    ref_path = PROJECT_ROOT / 'data' / 'test_split_LJ13-1000.npy'
    assert ref_path.exists(), f"Reference data not found: {ref_path}"
    ref_samples = np.load(ref_path, allow_pickle=True)
    print(f"Reference samples: {ref_samples.shape}")

    results = {}

    # 1. Baseline: lj13_asbs
    baseline_dir = PROJECT_ROOT / 'results' / 'lj13_asbs' / 'seed_0'
    results['lj13_baseline'], _ = evaluate_checkpoint(
        "LJ13 Baseline (ASBS)",
        baseline_dir / 'config.yaml',
        baseline_dir / 'checkpoints' / 'checkpoint_latest.pt',
        ref_samples,
    )

    # 2. KSD-ASBS checkpoint 50 (early)
    ksd_dir = PROJECT_ROOT / 'results' / 'lj13_ksd_asbs' / 'seed_0'
    results['lj13_ksd_ckpt50'], _ = evaluate_checkpoint(
        "LJ13 KSD-ASBS (checkpoint_50)",
        ksd_dir / 'config.yaml',
        ksd_dir / 'checkpoints' / 'checkpoint_50.pt',
        ref_samples,
    )

    # 3. KSD-ASBS checkpoint 1550 (late, before NaN)
    results['lj13_ksd_ckpt1550'], _ = evaluate_checkpoint(
        "LJ13 KSD-ASBS (checkpoint_1550)",
        ksd_dir / 'config.yaml',
        ksd_dir / 'checkpoints' / 'checkpoint_1550.pt',
        ref_samples,
    )

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Method':<35} {'energy_W2':>12} {'eq_W2':>12} {'dist_W2':>12} {'KSD²':>12} {'mean_E':>12}"
    print(header)
    print('-' * len(header))
    for name, agg in results.items():
        def fmt(key):
            if key in agg:
                return f"{agg[key]['mean']:.4f}±{agg[key]['std']:.4f}"
            return "N/A"
        print(f"{name:<35} {fmt('energy_w2'):>12} {fmt('eq_w2'):>12} {fmt('dist_w2'):>12} {fmt('ksd_squared'):>12} {fmt('mean_energy'):>12}")

    # Save results
    out_path = PROJECT_ROOT / 'evaluation' / 'eval_results_lj13.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
