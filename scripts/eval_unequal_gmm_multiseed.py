"""
scripts/eval_unequal_gmm_multiseed.py

Evaluates all valid unequal-weight GMM seeds:
  - ASBS (Baseline): seed 1, seed 2  (seed 0 diverged to NaN)
  - KSD-ASBS (λ=1.0): ALL diverged — skip
  - KSD-ASBS (λ=5.0): seed 0, seed 1, seed 2

Prints per-seed and aggregated metrics for RESULTS_v2.md
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


RESULTS_DIR = Path('/home/sky/SML/Stein_ASBS/results')
EVAL_DIR = Path('/home/sky/SML/Stein_ASBS/evaluation')
FIG_DIR = EVAL_DIR / 'figures_2d'


def load_model(exp_dir, device, ckpt_override=None):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / 'config.yaml'
    if ckpt_override:
        ckpt_path = Path(ckpt_override)
    else:
        ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    ts_cfg = {
        't0': float(cfg.timesteps.t0),
        't1': float(cfg.timesteps.t1),
        'steps': int(cfg.timesteps.steps),
        'rescale_t': cfg.timesteps.rescale_t if cfg.timesteps.rescale_t is not None else None,
    }
    return sde, source, energy, ts_cfg


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device):
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


def evaluate_unequal_gmm(samples, energy):
    # Move everything to CPU for safe metric computation
    samples_cpu = samples.cpu()
    centers = energy.get_centers().cpu()
    true_weights = energy.get_weights().cpu()
    std = energy.get_std()
    N = samples_cpu.shape[0]

    dists = torch.cdist(samples_cpu, centers)
    assignments = dists.argmin(dim=1)

    empirical_weights = torch.zeros(5)
    for k in range(5):
        empirical_weights[k] = (assignments == k).float().mean()

    counts = torch.zeros(5, dtype=torch.long)
    for k in range(5):
        counts[k] = (assignments == k).sum()

    min_dists = dists.min(dim=1).values
    threshold = 3.0 * std
    covered_mask = torch.zeros(5, dtype=torch.bool)
    for k in range(5):
        mode_samples = (assignments == k)
        if mode_samples.any():
            if min_dists[mode_samples].min() < threshold:
                covered_mask[k] = True
    n_covered = covered_mask.sum().item()

    E = energy.eval(samples_cpu)

    # Wasserstein-2 against reference
    torch.manual_seed(123)
    ref_samples = generate_reference_samples(energy, N, device='cpu')
    w2 = compute_w2(samples_cpu, ref_samples)

    results = {
        'empirical_weights': empirical_weights.cpu().tolist(),
        'true_weights': true_weights.cpu().tolist(),
        'weight_TV': 0.5 * (empirical_weights - true_weights).abs().sum().item(),
        'weight_KL': (true_weights * torch.log(true_weights / (empirical_weights + 1e-10))).sum().item(),
        'minority_mode_alive': bool(empirical_weights[4] > 0),
        'minority_mode_weight': empirical_weights[4].item(),
        'n_modes_covered': n_covered,
        'n_modes_total': 5,
        'per_mode_counts': counts.cpu().tolist(),
        'mean_energy': E.mean().item(),
        'std_energy': E.std().item(),
        'W2': w2,
    }
    return results


def generate_reference_samples(energy, n=2000, device='cpu'):
    weights = energy.get_weights().cpu()
    centers = energy.get_centers().cpu()
    std = energy.get_std()
    mode_idx = torch.multinomial(weights, n, replacement=True)
    samples = centers[mode_idx] + std * torch.randn(n, 2)
    return samples.to(device)


def compute_w2(samples, ref_samples):
    """Approximate W2 via sorting along each dimension."""
    d = samples.shape[1]
    w2_sq = 0.0
    for dim in range(d):
        s_sorted = samples[:, dim].sort().values
        r_sorted = ref_samples[:, dim].sort().values
        w2_sq += ((s_sorted - r_sorted) ** 2).mean().item()
    return w2_sq ** 0.5


def eval_seed(exp_dir, seed, device, n_samples=2000):
    """Evaluate a single seed. Returns metrics dict or None if failed."""
    seed_dir = RESULTS_DIR / exp_dir / f'seed_{seed}'
    if not seed_dir.exists():
        print(f"  [SKIP] {seed_dir} does not exist")
        return None
    ckpt = seed_dir / 'checkpoints' / 'checkpoint_latest.pt'
    if not ckpt.exists():
        print(f"  [SKIP] No checkpoint at {ckpt}")
        return None

    try:
        sde, source, energy, ts_cfg = load_model(seed_dir, device)
        torch.manual_seed(seed * 1000 + 42)
        samples = generate_samples(sde, source, ts_cfg, n_samples, device)

        # Check for NaN samples
        if torch.isnan(samples).any():
            print(f"  [SKIP] {exp_dir}/seed_{seed} produces NaN samples")
            return None

        metrics = evaluate_unequal_gmm(samples, energy)
        return metrics
    except Exception as e:
        print(f"  [ERROR] {exp_dir}/seed_{seed}: {e}")
        return None


def aggregate_metrics(all_metrics):
    """Compute mean +/- std across seeds."""
    keys = ['weight_TV', 'weight_KL', 'W2', 'mean_energy', 'minority_mode_weight']
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {'mean': np.mean(vals), 'std': np.std(vals)}
    # Mode coverage - take min
    agg['n_modes_covered'] = {
        'mean': np.mean([m['n_modes_covered'] for m in all_metrics]),
        'std': np.std([m['n_modes_covered'] for m in all_metrics]),
    }
    return agg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"=== Unequal-Weight GMM — Multi-Seed Evaluation ===")
    print(f"Time: {kst}")
    print(f"Device: {device}")
    print(f"N samples: 2000")

    # Define valid runs
    configs = [
        ('unequal_gmm_asbs', 'ASBS', [1, 2]),  # seed 0 diverged
        # KSD lambda=1: ALL diverged, skip entirely
        ('unequal_gmm_ksd_asbs_lam5', 'KSD λ=5', [0, 1, 2]),
    ]

    all_results = {}

    for exp_dir, label, seeds in configs:
        print(f"\n--- {label} ({exp_dir}) ---")
        seed_metrics = []
        for s in seeds:
            print(f"  Evaluating seed {s}...")
            m = eval_seed(exp_dir, s, device)
            if m is not None:
                seed_metrics.append((s, m))
                print(f"    Modes: {m['n_modes_covered']}/5 | W2: {m['W2']:.4f} | "
                      f"TV: {m['weight_TV']:.4f} | KL: {m['weight_KL']:.4f} | "
                      f"Minority: {m['minority_mode_weight']:.4f} | "
                      f"E_mean: {m['mean_energy']:.4f}")
                print(f"    Weights: {[f'{w:.4f}' for w in m['empirical_weights']]}")
                print(f"    True:    {[f'{w:.4f}' for w in m['true_weights']]}")
                print(f"    Counts:  {m['per_mode_counts']}")

        if seed_metrics:
            agg = aggregate_metrics([m for _, m in seed_metrics])
            all_results[label] = {
                'seeds': seed_metrics,
                'agg': agg,
                'exp_dir': exp_dir,
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (for RESULTS_v2.md)")
    print("=" * 80)

    # Per-seed table
    print("\n### Per-Seed Metrics\n")
    print("| Experiment | Modes | W2 | Weight TV | Weight KL | Minority (3%) | E_mean |")
    print("|---|---|---|---|---|---|---|")
    for label, data in all_results.items():
        for s, m in data['seeds']:
            print(f"| {label} s{s} | {m['n_modes_covered']}/5 | "
                  f"{m['W2']:.3f} | {m['weight_TV']:.4f} | {m['weight_KL']:.4f} | "
                  f"{m['minority_mode_weight']:.4f} | {m['mean_energy']:.4f} |")

    # Aggregated table
    print("\n### Aggregated (mean +/- std across seeds)\n")
    print("| Method | W2 | Weight TV | Weight KL | Minority wt |")
    print("|---|---|---|---|---|")
    for label, data in all_results.items():
        a = data['agg']
        n = len(data['seeds'])
        print(f"| {label} ({n} seeds) | "
              f"{a['W2']['mean']:.3f} +/- {a['W2']['std']:.3f} | "
              f"{a['weight_TV']['mean']:.4f} +/- {a['weight_TV']['std']:.4f} | "
              f"{a['weight_KL']['mean']:.4f} +/- {a['weight_KL']['std']:.4f} | "
              f"{a['minority_mode_weight']['mean']:.4f} +/- {a['minority_mode_weight']['std']:.4f} |")

    print("\n=== Done! ===")


if __name__ == '__main__':
    main()
