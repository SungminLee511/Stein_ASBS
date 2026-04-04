"""
evaluate_muller.py

Evaluate all Müller-Brown experiments (baseline ASBS vs KSD-ASBS, 3 seeds each).
Generates:
  - Per-seed metrics (energy_W2, KSD², mean energy, mode counts)
  - Aggregate statistics across seeds
  - 2D visualization: reference vs baseline vs KSD-ASBS samples on contour
  - JSON results for RESULTS.md

Usage:
    cd /home/RESEARCH/Stein_ASBS
    conda run -n Sampling_env python -u evaluation/evaluate_muller.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import OmegaConf
import ot as pot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Mode counting for Müller-Brown
# ====================================================================

MULLER_MINIMA = [
    {"name": "Min A (deepest)", "loc": (-0.558, 1.442), "E_approx": -146.7},
    {"name": "Min B",           "loc": (0.623, 0.028),  "E_approx": -108.2},
    {"name": "Min C (shallowest)", "loc": (-0.050, 0.467), "E_approx": -80.8},
]

def count_muller_modes(samples: torch.Tensor, radius: float = 0.3) -> dict:
    """Count how many of the 3 Müller-Brown minima have samples nearby."""
    s = samples.cpu().numpy()
    mode_counts = []
    modes_covered = 0
    for m in MULLER_MINIMA:
        loc = np.array(m["loc"])
        dists = np.linalg.norm(s - loc, axis=1)
        count = int((dists < radius).sum())
        mode_counts.append({"name": m["name"], "loc": m["loc"], "count": count})
        if count > 0:
            modes_covered += 1
    return {
        "modes_covered": modes_covered,
        "modes_total": 3,
        "per_mode": mode_counts,
    }


# ====================================================================
# Load and sample
# ====================================================================

def load_and_sample(exp_dir: Path, device: str, n_samples: int = 2000):
    """Load checkpoint and generate terminal samples."""
    cfg_path = exp_dir / 'config.yaml'
    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'

    if not cfg_path.exists() or not ckpt_path.exists():
        print(f"  SKIP: missing config or checkpoint in {exp_dir}")
        return None, None, None

    cfg = OmegaConf.load(cfg_path)

    # Resolve interpolations manually
    nfe = cfg.get('nfe', 100)
    rescale_t = cfg.get('rescale_t', None)
    dim = cfg.get('dim', 2)
    scale = cfg.get('scale', 2)
    sigma_max = cfg.get('sigma_max', 3)
    sigma_min = cfg.get('sigma_min', 0.01)

    # Instantiate energy
    energy = hydra.utils.instantiate(cfg.energy, device=device)

    # Source
    source = hydra.utils.instantiate(cfg.source, device=device)

    # SDE + controller
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    # Generate samples
    ts_cfg = {
        't0': float(cfg.timesteps.t0),
        't1': float(cfg.timesteps.t1),
        'steps': nfe,
        'rescale_t': rescale_t,
    }

    all_samples = []
    batch_size = min(n_samples, 2000)
    n = 0
    with torch.no_grad():
        while n < n_samples:
            b = min(batch_size, n_samples - n)
            x0 = source.sample([b]).to(device)
            ts = train_utils.get_timesteps(**ts_cfg).to(device)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            all_samples.append(x1)
            n += b
    samples = torch.cat(all_samples)[:n_samples]

    return samples, energy, cfg


def compute_metrics(samples, energy, ref_samples, device):
    """Compute all metrics for muller samples."""
    metrics = {}
    N = samples.shape[0]

    # Energy statistics (no grad needed)
    with torch.no_grad():
        gen_E = energy.eval(samples)
        metrics['mean_energy'] = gen_E.mean().item()
        metrics['std_energy'] = gen_E.std().item()
        metrics['min_energy'] = gen_E.min().item()

        # Raw energies (unscaled) for interpretability
        raw_E = energy._eval_raw(samples)
        metrics['mean_raw_energy'] = raw_E.mean().item()
        metrics['min_raw_energy'] = raw_E.min().item()

    # KSD² — needs autograd for scores
    samples_req = samples.detach().clone().requires_grad_(True)
    E = energy.eval(samples_req)
    scores = -torch.autograd.grad(E.sum(), samples_req)[0]

    with torch.no_grad():
        ell = median_bandwidth(samples)
        N_ksd = min(N, 2000)
        idx = torch.randperm(N, device=device)[:N_ksd]
        metrics['ksd_squared'] = compute_ksd_squared(
            samples[idx], scores[idx].detach(), ell
        ).item()

    # Energy W2 vs reference
    with torch.no_grad():
        if ref_samples is not None:
            ref = ref_samples.to(device)
            B = min(N, len(ref))
            idx_g = torch.randperm(N, device=device)[:B]
            idx_r = torch.randperm(len(ref), device=device)[:B]
            ref_E = energy.eval(ref[idx_r])
            metrics['energy_w2'] = float(
                pot.emd2_1d(ref_E.cpu().numpy(), gen_E[idx_g].cpu().numpy()) ** 0.5
            )

    # Mode coverage
    mode_info = count_muller_modes(samples, radius=0.3)
    metrics['modes_covered'] = mode_info['modes_covered']
    metrics['per_mode'] = mode_info['per_mode']

    return metrics


# ====================================================================
# Visualization
# ====================================================================

def plot_muller_comparison(energy, ref_samples, baseline_samples_dict, ksd_samples_dict, save_path):
    """Create side-by-side 2D contour plots: ref, baseline (best seed), KSD (best seed)."""
    # Energy landscape
    x = torch.linspace(-1.5, 1.2, 300)
    y = torch.linspace(-0.5, 2.0, 300)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    E = energy._eval_raw(grid).reshape(300, 300).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax in axes:
        cs = ax.contourf(xx.numpy(), yy.numpy(), E, levels=50, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        # Mark minima
        for m in MULLER_MINIMA:
            ax.plot(m['loc'][0], m['loc'][1], 'w*', markersize=12, markeredgecolor='black')

    # Panel 1: Reference
    axes[0].set_title('Reference (rejection sampling)', fontsize=13)
    ref = ref_samples[:1000].numpy()
    axes[0].scatter(ref[:, 0], ref[:, 1], s=2, c='white', alpha=0.4)

    # Panel 2: Baseline ASBS (use seed with most modes, then lowest energy_w2)
    axes[1].set_title('Baseline ASBS', fontsize=13)
    if baseline_samples_dict:
        # Pick best seed
        best_seed = max(baseline_samples_dict.keys(),
                       key=lambda s: baseline_samples_dict[s].shape[0])
        s = baseline_samples_dict[best_seed][:1000].cpu().numpy()
        axes[1].scatter(s[:, 0], s[:, 1], s=3, c='red', alpha=0.5)
        mode_info = count_muller_modes(baseline_samples_dict[best_seed])
        axes[1].set_title(f'Baseline ASBS (seed {best_seed})\n'
                          f'Modes: {mode_info["modes_covered"]}/3', fontsize=12)

    # Panel 3: KSD-ASBS
    axes[2].set_title('KSD-ASBS', fontsize=13)
    if ksd_samples_dict:
        best_seed = max(ksd_samples_dict.keys(),
                       key=lambda s: ksd_samples_dict[s].shape[0])
        s = ksd_samples_dict[best_seed][:1000].cpu().numpy()
        axes[2].scatter(s[:, 0], s[:, 1], s=3, c='orange', alpha=0.5)
        mode_info = count_muller_modes(ksd_samples_dict[best_seed])
        axes[2].set_title(f'KSD-ASBS (seed {best_seed})\n'
                          f'Modes: {mode_info["modes_covered"]}/3', fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {save_path}")


def plot_all_seeds(energy, baseline_all, ksd_all, save_path):
    """Plot all seeds side by side: 2 rows × 3 cols."""
    x = torch.linspace(-1.5, 1.2, 300)
    y = torch.linspace(-0.5, 2.0, 300)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    E = energy._eval_raw(grid).reshape(300, 300).numpy()

    seeds = sorted(set(list(baseline_all.keys()) + list(ksd_all.keys())))
    n_seeds = len(seeds)

    fig, axes = plt.subplots(2, n_seeds, figsize=(6 * n_seeds, 10))

    for row in range(2):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            ax.contourf(xx.numpy(), yy.numpy(), E, levels=50, cmap='viridis', alpha=0.8)
            for m in MULLER_MINIMA:
                ax.plot(m['loc'][0], m['loc'][1], 'w*', markersize=10, markeredgecolor='black')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            if row == 0:  # Baseline
                if seed in baseline_all:
                    s = baseline_all[seed][:1000].cpu().numpy()
                    ax.scatter(s[:, 0], s[:, 1], s=3, c='red', alpha=0.5)
                    mi = count_muller_modes(baseline_all[seed])
                    ax.set_title(f'Baseline seed={seed}\nModes: {mi["modes_covered"]}/3')
                else:
                    ax.set_title(f'Baseline seed={seed}\n(no data)')
            else:  # KSD
                if seed in ksd_all:
                    s = ksd_all[seed][:1000].cpu().numpy()
                    ax.scatter(s[:, 0], s[:, 1], s=3, c='orange', alpha=0.5)
                    mi = count_muller_modes(ksd_all[seed])
                    ax.set_title(f'KSD-ASBS seed={seed}\nModes: {mi["modes_covered"]}/3')
                else:
                    ax.set_title(f'KSD-ASBS seed={seed}\n(no data)')

    fig.suptitle('Müller-Brown: All Seeds Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all-seeds plot: {save_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results_root = Path('/home/RESEARCH/Stein_ASBS/results')
    eval_dir = Path('/home/RESEARCH/Stein_ASBS/evaluation')
    fig_dir = eval_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 2000
    n_eval_seeds = 5  # Number of sampling seeds per checkpoint

    experiments = {
        'muller_asbs': results_root / 'muller_asbs',
        'muller_ksd_asbs': results_root / 'muller_ksd_asbs',
    }

    all_results = {}
    all_samples = {'muller_asbs': {}, 'muller_ksd_asbs': {}}
    energy_ref = None  # Will be set on first load

    for exp_name, exp_dir in experiments.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        seed_dirs = sorted(exp_dir.glob('seed_*'))
        if not seed_dirs:
            print(f"  No seed directories found in {exp_dir}")
            continue

        exp_results = {}
        for seed_dir in seed_dirs:
            seed_num = int(seed_dir.name.split('_')[1])
            print(f"\n  --- {exp_name} / seed {seed_num} ---")

            # Multiple evaluation sampling seeds
            seed_eval_metrics = []
            best_samples = None

            for eval_seed in range(n_eval_seeds):
                torch.manual_seed(eval_seed * 7777 + seed_num * 42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed * 7777 + seed_num * 42)

                samples, energy, cfg = load_and_sample(seed_dir, device, n_samples)
                if samples is None:
                    break

                if energy_ref is None:
                    energy_ref = energy

                ref_samples = energy.get_ref_samples()
                metrics = compute_metrics(samples, energy, ref_samples, device)
                seed_eval_metrics.append(metrics)

                if eval_seed == 0:
                    best_samples = samples.clone()

            if not seed_eval_metrics:
                continue

            # Store samples for visualization (eval_seed=0)
            all_samples[exp_name][seed_num] = best_samples

            # Aggregate across eval seeds
            agg = {}
            scalar_keys = ['mean_energy', 'std_energy', 'min_energy', 'mean_raw_energy',
                          'min_raw_energy', 'ksd_squared', 'energy_w2', 'modes_covered']
            for key in scalar_keys:
                vals = [m[key] for m in seed_eval_metrics if key in m]
                if vals:
                    agg[key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'values': vals,
                    }
            # Per-mode info from first eval seed
            agg['per_mode'] = seed_eval_metrics[0].get('per_mode', [])

            exp_results[f'seed_{seed_num}'] = agg
            print(f"    energy_W2: {agg.get('energy_w2', {}).get('mean', 'N/A'):.4f} "
                  f"± {agg.get('energy_w2', {}).get('std', 'N/A'):.4f}")
            print(f"    KSD²: {agg.get('ksd_squared', {}).get('mean', 'N/A'):.6f}")
            print(f"    modes_covered: {agg.get('modes_covered', {}).get('mean', 'N/A'):.1f}/3")
            print(f"    mean_raw_energy: {agg.get('mean_raw_energy', {}).get('mean', 'N/A'):.1f}")

        all_results[exp_name] = exp_results

    # ---- Aggregate across training seeds ----
    print(f"\n{'='*60}")
    print("Aggregate Results (across training seeds)")
    print(f"{'='*60}")

    summary = {}
    for exp_name in ['muller_asbs', 'muller_ksd_asbs']:
        if exp_name not in all_results:
            continue
        method_agg = defaultdict(list)
        for seed_key, seed_data in all_results[exp_name].items():
            for metric_key, metric_val in seed_data.items():
                if isinstance(metric_val, dict) and 'mean' in metric_val:
                    method_agg[metric_key].append(metric_val['mean'])

        method_summary = {}
        for metric_key, vals in method_agg.items():
            method_summary[metric_key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'per_seed_means': vals,
            }
        summary[exp_name] = method_summary

        print(f"\n  {exp_name}:")
        for k, v in method_summary.items():
            print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # ---- Save JSON ----
    json_path = eval_dir / 'muller_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'per_seed': all_results,
            'summary': summary,
        }, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    # ---- Visualizations ----
    if energy_ref is not None:
        ref_samples = energy_ref.get_ref_samples()

        # Main comparison plot
        plot_muller_comparison(
            energy_ref, ref_samples,
            all_samples.get('muller_asbs', {}),
            all_samples.get('muller_ksd_asbs', {}),
            fig_dir / 'muller_comparison.png'
        )

        # All seeds plot
        plot_all_seeds(
            energy_ref,
            all_samples.get('muller_asbs', {}),
            all_samples.get('muller_ksd_asbs', {}),
            fig_dir / 'muller_all_seeds.png'
        )

    print("\n=== Müller-Brown evaluation complete! ===")


if __name__ == '__main__':
    main()
