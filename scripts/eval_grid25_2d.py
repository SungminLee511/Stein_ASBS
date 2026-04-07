"""
scripts/eval_grid25_2d.py

Evaluate Grid25 using the same figure style as eval_2d_all.py (GMM9 style):
  - 3-panel terminal: Ground Truth | ASBS | KSD-ASBS
  - 5-panel marginal evolution per method
  - Append to 2d_result.md
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


RESULTS_DIR = Path('/home/RESEARCH/Stein_ASBS/results')
EVAL_DIR = Path('/home/RESEARCH/Stein_ASBS/evaluation')
FIG_DIR = EVAL_DIR / 'figures_2d'

XLIM = (-6, 6)
YLIM = (-6, 6)


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


@torch.no_grad()
def generate_full_states(sde, source, ts_cfg, n_samples, device):
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    states = sdeint(sde, x0, ts, only_boundary=False)
    return states, ts


def assign_modes(samples, centers, threshold_factor=3.0, std=0.3):
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)
    min_dists = dists.min(dim=1).values
    threshold = threshold_factor * std
    assignments = nearest.clone()
    assignments[min_dists > threshold] = -1
    K = centers.shape[0]
    counts = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        counts[k] = (assignments == k).sum()
    n_covered = (counts > 0).sum().item()
    return assignments, counts, n_covered


def compute_metrics(samples, energy, centers, std):
    assignments, counts, n_covered = assign_modes(
        samples, centers.to(samples.device), std=std
    )
    E = energy.eval(samples)
    return {
        'n_modes_covered': n_covered,
        'n_modes_total': centers.shape[0],
        'per_mode_counts': counts.cpu().tolist(),
        'mean_energy': E.mean().item(),
        'std_energy': E.std().item(),
    }


# ====================================================================
# Plotting — same style as eval_2d_all.py
# ====================================================================

def plot_terminal(
    energy, samples_ref, samples_base, samples_ksd,
    centers, m_base, m_ksd,
    benchmark_name, output_path,
):
    """3-panel: Ground Truth | Baseline ASBS | KSD-ASBS"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    titles = [
        'Ground Truth',
        f'ASBS\n({m_base["n_modes_covered"]}/{m_base["n_modes_total"]} modes)',
        f'KSD-ASBS\n({m_ksd["n_modes_covered"]}/{m_ksd["n_modes_total"]} modes)',
    ]

    sample_sets = [samples_ref, samples_base, samples_ksd]
    colors = ['gray', '#d62728', '#ff7f0e']

    for idx, (ax, title, samples, color) in enumerate(zip(axes, titles, sample_sets, colors)):
        ax.set_facecolor('#f7f7f7')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')

        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=80, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=4, c=color, alpha=0.4, zorder=5)

    fig.suptitle(f'{benchmark_name}: Terminal Distribution', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_marginal_evolution(
    energy, states, ts, centers, std,
    method_name, benchmark_name, output_path,
    n_snapshots=5,
):
    """5-panel figure showing marginal distribution at evenly spaced timesteps."""
    T = len(states)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5.5))

    color = '#d62728' if method_name == 'ASBS' else '#ff7f0e'

    for panel_idx, state_idx in enumerate(indices):
        ax = axes[panel_idx]
        t_val = ts[state_idx].item()
        samples = states[state_idx].cpu().numpy()

        ax.scatter(samples[:, 0], samples[:, 1], s=4, c=color, alpha=0.4, zorder=5)

        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=80, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        ax.set_facecolor('#f7f7f7')
        ax.set_title(f't = {t_val:.2f}', fontsize=13, fontweight='bold')

    fig.suptitle(f'{benchmark_name}: {method_name} — Marginal Evolution', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_dir = RESULTS_DIR / 'grid25_asbs' / 'seed_0'
    ksd_dir = RESULTS_DIR / 'grid25_ksd_asbs_lam01' / 'seed_0'
    bench_name = '25-Mode Grid (5×5)'
    ksd_lambda = 0.1

    n_samples = 2000

    print("=" * 60)
    print(f"  Evaluating: {bench_name}")
    print("=" * 60)

    print("  Loading baseline ASBS...")
    sde_base, src_base, energy, ts_base = load_model(baseline_dir, device)
    print("  Loading KSD-ASBS...")
    sde_ksd, src_ksd, _, ts_ksd = load_model(ksd_dir, device)

    centers = energy.get_centers().to(device)
    std = energy.get_std()

    # Terminal samples
    print(f"  Generating {n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_ksd = generate_samples(sde_ksd, src_ksd, ts_ksd, n_samples, device)
    samples_ref = energy.get_ref_samples(n_samples).to(device)

    # Metrics
    print("  Computing metrics...")
    m_base = compute_metrics(samples_base, energy, centers, std)
    m_ksd = compute_metrics(samples_ksd, energy, centers, std)

    K = centers.shape[0]
    print(f"    ASBS:     {m_base['n_modes_covered']}/{K} modes, mean_E={m_base['mean_energy']:.4f}")
    print(f"    KSD-ASBS: {m_ksd['n_modes_covered']}/{K} modes, mean_E={m_ksd['mean_energy']:.4f}")
    print(f"    ASBS per-mode:     {m_base['per_mode_counts']}")
    print(f"    KSD-ASBS per-mode: {m_ksd['per_mode_counts']}")

    # 3-panel terminal
    print("  Generating terminal distribution figure (3-panel)...")
    plot_terminal(
        energy, samples_ref, samples_base, samples_ksd,
        centers, m_base, m_ksd,
        bench_name,
        FIG_DIR / 'grid25_terminal.png',
    )

    # Full trajectories for marginal evolution
    print("  Generating full trajectories for marginal snapshots...")
    torch.manual_seed(42)
    states_base, ts_base_full = generate_full_states(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(42)
    states_ksd, ts_ksd_full = generate_full_states(sde_ksd, src_ksd, ts_ksd, n_samples, device)

    # 5-panel marginal ASBS
    print("  Generating ASBS marginal evolution figure...")
    plot_marginal_evolution(
        energy, states_base, ts_base_full, centers, std,
        'ASBS', bench_name,
        FIG_DIR / 'grid25_marginal_asbs.png',
    )

    # 5-panel marginal KSD-ASBS
    print("  Generating KSD-ASBS marginal evolution figure...")
    plot_marginal_evolution(
        energy, states_ksd, ts_ksd_full, centers, std,
        'KSD-ASBS', bench_name,
        FIG_DIR / 'grid25_marginal_ksd.png',
    )

    # Append to 2d_result.md
    print("  Appending to 2d_result.md...")
    md_path = EVAL_DIR / '2d_result.md'

    section = f"""## {bench_name}

| Metric | ASBS (Baseline) | KSD-ASBS (lambda={ksd_lambda}) |
|---|---|---|
| Modes covered (of {K}) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} |
| Std energy | {m_base['std_energy']:.4f} | {m_ksd['std_energy']:.4f} |
| Per-mode counts | {m_base['per_mode_counts']} | {m_ksd['per_mode_counts']} |

### Terminal Distribution

![grid25 terminal](figures_2d/grid25_terminal.png)

### Marginal Evolution: ASBS

![grid25 marginal asbs](figures_2d/grid25_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![grid25 marginal ksd](figures_2d/grid25_marginal_ksd.png)

---

"""

    with open(md_path, 'a') as f:
        f.write(section)

    print(f"  Appended Grid25 results to {md_path}")
    print("\n=== Grid25 evaluation complete! ===")


if __name__ == '__main__':
    main()
