"""
scripts/eval_grid25.py

Evaluate Grid25 (5x5 Gaussian grid) benchmark:
  - Figure A: 4-panel terminal distribution (GT, ASBS, KSD-ASBS, Overlay)
  - Figure B: 3-panel SDE trajectories (GT, ASBS trajectories, KSD trajectories)
  - Metrics table
  - Append results to RESULTS.md
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


# ====================================================================
# Loading
# ====================================================================

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


# ====================================================================
# Sampling
# ====================================================================

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


# ====================================================================
# Metrics
# ====================================================================

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
# Plotting: density contours
# ====================================================================

def plot_density_contours(ax, energy, xlim, ylim, n_grid=200):
    x = torch.linspace(xlim[0], xlim[1], n_grid)
    y = torch.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        E = energy.eval(grid).reshape(n_grid, n_grid)
    log_p = -E
    log_p = log_p - log_p.max()
    p = torch.exp(log_p)
    ax.contourf(xx.numpy(), yy.numpy(), p.numpy(), levels=30, cmap='Blues', alpha=0.6)
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(), levels=10, colors='steelblue',
               alpha=0.4, linewidths=0.5)


# ====================================================================
# Figure A: Terminal Distribution (4-panel)
# ====================================================================

def plot_terminal_4panel(
    energy, samples_ref, samples_base, samples_ksd,
    centers, m_base, m_ksd, output_path,
):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

    titles = [
        'Ground Truth',
        f'ASBS\n({m_base["n_modes_covered"]}/{m_base["n_modes_total"]} modes)',
        f'KSD-ASBS\n({m_ksd["n_modes_covered"]}/{m_ksd["n_modes_total"]} modes)',
        'Overlay (ASBS=red, KSD=orange)',
    ]

    for ax in axes:
        ax.set_facecolor('#f7f7f7')
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        # Plot density contours
        plot_density_contours(ax, energy, XLIM, YLIM)
        # Mode center stars
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=60, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Panel 0: Ground truth
    axes[0].set_title(titles[0], fontsize=13, fontweight='bold')
    s = samples_ref.cpu().numpy()
    axes[0].scatter(s[:, 0], s[:, 1], s=3, c='gray', alpha=0.4, zorder=5)

    # Panel 1: ASBS
    axes[1].set_title(titles[1], fontsize=13, fontweight='bold')
    s = samples_base.cpu().numpy()
    axes[1].scatter(s[:, 0], s[:, 1], s=3, c='#d62728', alpha=0.4, zorder=5)

    # Panel 2: KSD-ASBS
    axes[2].set_title(titles[2], fontsize=13, fontweight='bold')
    s = samples_ksd.cpu().numpy()
    axes[2].scatter(s[:, 0], s[:, 1], s=3, c='#ff7f0e', alpha=0.4, zorder=5)

    # Panel 3: Overlay
    axes[3].set_title(titles[3], fontsize=13, fontweight='bold')
    s_b = samples_base.cpu().numpy()
    s_k = samples_ksd.cpu().numpy()
    axes[3].scatter(s_b[:, 0], s_b[:, 1], s=3, c='#d62728', alpha=0.3, zorder=5, label='ASBS')
    axes[3].scatter(s_k[:, 0], s_k[:, 1], s=3, c='#ff7f0e', alpha=0.3, zorder=6, label='KSD-ASBS')
    axes[3].legend(fontsize=10, loc='upper right')

    fig.suptitle('25-Mode Grid (5×5): Terminal Distribution', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Figure B: SDE Trajectories (3-panel)
# ====================================================================

MODE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
]


def plot_trajectories(
    energy, states, ts, centers, std,
    method_name, ax, n_traj=80, step_skip=5,
):
    """Plot trajectories colored by terminal mode assignment."""
    T = len(states)
    n_samples = states[0].shape[0]

    # Terminal samples
    terminal = states[-1]  # (N, 2)

    # Assign terminal samples to modes
    assignments, _, _ = assign_modes(terminal, centers.to(terminal.device), std=std)

    # Select n_traj trajectories
    indices = list(range(min(n_traj, n_samples)))

    # Time indices to plot (every step_skip)
    t_indices = list(range(0, T, step_skip))
    if t_indices[-1] != T - 1:
        t_indices.append(T - 1)

    for i in indices:
        mode_id = assignments[i].item()
        if mode_id == -1:
            color = '#cccccc'
            alpha = 0.15
        else:
            color = MODE_COLORS[mode_id % len(MODE_COLORS)]
            alpha = 0.4

        traj_x = [states[t][i, 0].cpu().item() for t in t_indices]
        traj_y = [states[t][i, 1].cpu().item() for t in t_indices]
        ax.plot(traj_x, traj_y, color=color, alpha=alpha, linewidth=0.6, zorder=3)

    # Terminal scatter on top
    for i in indices:
        mode_id = assignments[i].item()
        if mode_id == -1:
            color = '#cccccc'
        else:
            color = MODE_COLORS[mode_id % len(MODE_COLORS)]
        x_end = terminal[i, 0].cpu().item()
        y_end = terminal[i, 1].cpu().item()
        ax.scatter(x_end, y_end, s=8, c=color, zorder=7, edgecolors='none')


def plot_trajectories_3panel(
    energy, samples_ref, states_base, ts_base,
    states_ksd, ts_ksd, centers, std, output_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    titles = [
        'Ground Truth',
        'ASBS — Trajectories',
        'KSD-ASBS — Trajectories',
    ]

    for ax in axes:
        ax.set_facecolor('#f7f7f7')
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        # Mode center stars
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=60, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Panel 0: Ground truth reference
    axes[0].set_title(titles[0], fontsize=13, fontweight='bold')
    plot_density_contours(axes[0], energy, XLIM, YLIM)
    s = samples_ref.cpu().numpy()
    axes[0].scatter(s[:, 0], s[:, 1], s=3, c='gray', alpha=0.4, zorder=5)

    # Panel 1: ASBS trajectories
    axes[1].set_title(titles[1], fontsize=13, fontweight='bold')
    plot_trajectories(energy, states_base, ts_base, centers, std, 'ASBS', axes[1])

    # Panel 2: KSD-ASBS trajectories
    axes[2].set_title(titles[2], fontsize=13, fontweight='bold')
    plot_trajectories(energy, states_ksd, ts_ksd, centers, std, 'KSD-ASBS', axes[2])

    fig.suptitle('25-Mode Grid (5×5): SDE Trajectories', fontsize=16, y=1.02)
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

    n_samples = 2000

    print("=" * 60)
    print("  Evaluating: 25-Mode Grid (5×5)")
    print("=" * 60)

    # Load models
    print("  Loading baseline ASBS...")
    sde_base, src_base, energy, ts_base = load_model(baseline_dir, device)
    print("  Loading KSD-ASBS...")
    sde_ksd, src_ksd, _, ts_ksd = load_model(ksd_dir, device)

    centers = energy.get_centers().to(device)
    std = energy.get_std()

    # --- Terminal samples ---
    print(f"  Generating {n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_ksd = generate_samples(sde_ksd, src_ksd, ts_ksd, n_samples, device)
    samples_ref = energy.get_ref_samples(n_samples).to(device)

    # --- Metrics ---
    print("  Computing metrics...")
    m_base = compute_metrics(samples_base, energy, centers, std)
    m_ksd = compute_metrics(samples_ksd, energy, centers, std)

    K = centers.shape[0]
    print(f"    ASBS:     {m_base['n_modes_covered']}/{K} modes, mean_E={m_base['mean_energy']:.4f}")
    print(f"    KSD-ASBS: {m_ksd['n_modes_covered']}/{K} modes, mean_E={m_ksd['mean_energy']:.4f}")
    print(f"    ASBS per-mode:     {m_base['per_mode_counts']}")
    print(f"    KSD-ASBS per-mode: {m_ksd['per_mode_counts']}")

    # --- Figure A: Terminal Distribution (4-panel) ---
    print("  Generating terminal distribution figure (4-panel)...")
    plot_terminal_4panel(
        energy, samples_ref, samples_base, samples_ksd,
        centers, m_base, m_ksd,
        FIG_DIR / 'grid25_terminal.png',
    )

    # --- Full trajectories ---
    print("  Generating full trajectories (80 particles)...")
    torch.manual_seed(42)
    states_base, ts_base_full = generate_full_states(sde_base, src_base, ts_base, 200, device)
    torch.manual_seed(42)
    states_ksd, ts_ksd_full = generate_full_states(sde_ksd, src_ksd, ts_ksd, 200, device)

    # --- Figure B: Trajectories (3-panel) ---
    print("  Generating trajectory figure (3-panel)...")
    plot_trajectories_3panel(
        energy, samples_ref, states_base, ts_base_full,
        states_ksd, ts_ksd_full, centers, std,
        FIG_DIR / 'grid25_trajectories.png',
    )

    # --- Append to RESULTS.md ---
    print("  Appending to RESULTS.md...")
    kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

    delta_modes = m_ksd['n_modes_covered'] - m_base['n_modes_covered']
    delta_sign = '+' if delta_modes >= 0 else ''

    section = f"""

### 2D Visualization: 25-Mode Grid (5×5)

Evaluated: {kst}

| Metric | Baseline ASBS | SDR-ASBS | Δ |
|---|---|---|---|
| Modes covered (of 25) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} | {delta_sign}{delta_modes} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} | {m_ksd['mean_energy'] - m_base['mean_energy']:+.4f} |
| Std energy | {m_base['std_energy']:.4f} | {m_ksd['std_energy']:.4f} | |
| Per-mode counts (base) | {m_base['per_mode_counts']} | | |
| Per-mode counts (SDR) | | {m_ksd['per_mode_counts']} | |

Terminal Distribution:

![grid25 terminal](figures_2d/grid25_terminal.png)

SDE Trajectories:

![grid25 trajectories](figures_2d/grid25_trajectories.png)

"""

    results_md = EVAL_DIR / 'RESULTS.md'
    with open(results_md, 'r') as f:
        content = f.read()
    with open(results_md, 'a') as f:
        f.write(section)

    print(f"  Appended Grid25 results to {results_md}")
    print("\n=== Grid25 evaluation complete! ===")


if __name__ == '__main__':
    main()
