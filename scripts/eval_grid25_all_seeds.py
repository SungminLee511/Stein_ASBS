"""
scripts/eval_grid25_all_seeds.py

Big 5-row NeurIPS figure: marginal evolution for all valid Grid25 experiments.
Rows: ASBS s0, ASBS s1, ASBS s2, KSD-ASBS s1, KSD-ASBS s2
Cols: 5 evenly-spaced timestep snapshots (t=0.00 ... t=1.00)

Also generates a 6-panel terminal distribution comparison
(Ground Truth + 5 experiments).
"""

import sys
sys.path.insert(0, '/home/sky/SML/Stein_ASBS')

import torch
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# NeurIPS style (from eval_grid25_neurips.py)
# ====================================================================

def set_neurips_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
    })


# ====================================================================
# Constants
# ====================================================================

RESULTS_DIR = Path('/home/sky/SML/Stein_ASBS/results')
FIG_DIR = Path('/home/sky/SML/Stein_ASBS/evaluation/figures_2d')

XLIM = (-6, 6)
YLIM = (-6, 6)

C_REF = '#555555'
C_ASBS = '#c0392b'       # deep red
C_SDR = '#e67e22'         # warm orange
C_CONTOUR = '#2c3e50'
C_BG = '#fafafa'

TEXT_WIDTH_IN = 5.5

EXPERIMENTS = [
    ('grid25_asbs/seed_0',     'ASBS (seed 0)',     C_ASBS),
    ('grid25_asbs/seed_1',     'ASBS (seed 1)',     C_ASBS),
    ('grid25_asbs/seed_2',     'ASBS (seed 2)',     C_ASBS),
    ('grid25_ksd_asbs/seed_1', 'SDR-ASBS (seed 1)', C_SDR),
    ('grid25_ksd_asbs/seed_2', 'SDR-ASBS (seed 2)', C_SDR),
]


# ====================================================================
# Loading & sampling (same as eval_grid25_neurips.py)
# ====================================================================

def load_model(exp_dir, device):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / 'config.yaml'
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


def plot_density_contours(ax, energy, xlim, ylim, n_grid=250):
    x = torch.linspace(xlim[0], xlim[1], n_grid)
    y = torch.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        E = energy.eval(grid).reshape(n_grid, n_grid)
    log_p = -E
    log_p = log_p - log_p.max()
    p = torch.exp(log_p)
    ax.contourf(xx.numpy(), yy.numpy(), p.numpy(), levels=25,
                cmap='Blues', alpha=0.5, zorder=1)
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(), levels=8,
               colors=C_CONTOUR, alpha=0.25, linewidths=0.3, zorder=2)


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


# ====================================================================
# Big Figure: 5-row marginal evolution
# ====================================================================

def plot_big_marginal(
    all_states, all_ts, all_labels, all_colors,
    energy, centers, output_path,
    n_snapshots=5,
):
    """5-row × 5-col grid: each row is one experiment's marginal evolution."""
    n_rows = len(all_states)
    n_cols = n_snapshots

    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95  # slightly less than square for tighter layout
    fig_w = TEXT_WIDTH_IN + 0.7  # extra for row labels
    fig_h = row_h * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    c_np = centers.cpu().numpy()

    for row, (states, ts, label, color) in enumerate(
        zip(all_states, all_ts, all_labels, all_colors)
    ):
        T = len(states)
        indices = np.linspace(0, T - 1, n_cols, dtype=int)

        for col, state_idx in enumerate(indices):
            ax = axes[row, col]
            t_val = ts[state_idx].item()
            samples = states[state_idx].cpu().numpy()

            ax.set_facecolor(C_BG)

            # Mode center markers
            ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=10, c='black',
                       linewidths=0.3, zorder=10)

            # Samples
            ax.scatter(samples[:, 0], samples[:, 1], s=1.0, c=color,
                       alpha=0.3, zorder=5, edgecolors='none', rasterized=True)

            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)
            ax.set_aspect('equal')

            # Column titles on top row only
            if row == 0:
                ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)

            # Tick cleanup
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.tick_params(length=2, width=0.4, labelsize=6)

            # Only show y-tick labels on leftmost column
            if col > 0:
                ax.set_yticklabels([])
            # Only show x-tick labels on bottom row
            if row < n_rows - 1:
                ax.set_xticklabels([])

        # Row label on left
        axes[row, 0].set_ylabel(label, fontsize=7.5, fontweight='bold',
                                labelpad=8)

    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path} + .pdf")


# ====================================================================
# Terminal distribution: 6-panel (GT + 5 experiments)
# ====================================================================

def plot_terminal_all(
    energy, ref_samples, all_terminal, all_labels, all_colors,
    all_mode_counts, centers, output_path,
):
    n_panels = 1 + len(all_terminal)  # GT + experiments
    n_cols = 3
    n_rows = 2
    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH_IN, row_h * n_rows))
    axes_flat = axes.flatten()

    c_np = centers.cpu().numpy()

    # Panel 0: Ground Truth
    ax = axes_flat[0]
    ax.set_facecolor(C_BG)
    plot_density_contours(ax, energy, XLIM, YLIM)
    ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=15, c='black',
               linewidths=0.5, zorder=10)
    s = ref_samples.cpu().numpy()
    ax.scatter(s[:, 0], s[:, 1], s=1.5, c=C_REF, alpha=0.35, zorder=5,
               edgecolors='none', rasterized=True)
    ax.set_title('Ground Truth', fontsize=9, pad=4)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(length=2, width=0.4)

    # Panels 1-5: experiments
    for i, (samples, label, color, n_cov) in enumerate(
        zip(all_terminal, all_labels, all_colors, all_mode_counts)
    ):
        ax = axes_flat[i + 1]
        ax.set_facecolor(C_BG)
        plot_density_contours(ax, energy, XLIM, YLIM)
        ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=15, c='black',
                   linewidths=0.5, zorder=10)
        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=1.5, c=color, alpha=0.35, zorder=5,
                   edgecolors='none', rasterized=True)
        ax.set_title(f'{label} ({n_cov}/25)', fontsize=8, pad=4)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(length=2, width=0.4)

    # Y-labels on left column
    for r in range(n_rows):
        axes[r, 0].set_ylabel(r'$x_2$', fontsize=9)
    for c_idx in range(n_cols):
        axes[-1, c_idx].set_xlabel(r'$x_1$', fontsize=9)

    fig.subplots_adjust(wspace=0.25, hspace=0.35)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path} + .pdf")


# ====================================================================
# Main
# ====================================================================

def main():
    set_neurips_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 2000

    print("=" * 60)
    print("  Grid25 — Big Picture (5 experiments)")
    print("=" * 60)

    energy = None
    centers = None
    std = None

    all_states = []
    all_ts = []
    all_terminal = []
    all_labels = []
    all_colors = []
    all_mode_counts = []

    for exp_rel, label, color in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_rel
        print(f"\n  Loading {label} from {exp_dir}...")

        sde, source, en, ts_cfg = load_model(exp_dir, device)
        if energy is None:
            energy = en
            centers = energy.get_centers().to(device)
            std = energy.get_std()

        # Full trajectories for marginal evolution
        torch.manual_seed(42)
        states, ts = generate_full_states(sde, source, ts_cfg, n_samples, device)
        all_states.append(states)
        all_ts.append(ts)

        # Terminal samples
        terminal = states[-1]
        all_terminal.append(terminal)
        all_labels.append(label)
        all_colors.append(color)

        # Mode coverage
        _, counts, n_covered = assign_modes(terminal, centers, std=std)
        all_mode_counts.append(n_covered)
        print(f"    Modes covered: {n_covered}/25")

        # Free SDE memory
        del sde, source
        torch.cuda.empty_cache()

    # Reference samples
    ref_samples = energy.get_ref_samples(n_samples).to(device)

    # ---- Figure 1: Big marginal evolution (5 rows × 5 cols) ----
    print("\n  Generating big marginal evolution figure...")
    plot_big_marginal(
        all_states, all_ts, all_labels, all_colors,
        energy, centers,
        FIG_DIR / 'grid25_all_seeds_marginal.png',
    )

    # ---- Figure 2: Terminal distribution (2×3 grid) ----
    print("  Generating terminal distribution figure...")
    plot_terminal_all(
        energy, ref_samples, all_terminal, all_labels, all_colors,
        all_mode_counts, centers,
        FIG_DIR / 'grid25_all_seeds_terminal.png',
    )

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print(f"  {'Experiment':<25} {'Modes':>10}")
    print("-" * 40)
    for label, n_cov in zip(all_labels, all_mode_counts):
        print(f"  {label:<25} {n_cov:>5}/25")
    print("=" * 60)
    print("\n  Done!")


if __name__ == '__main__':
    main()
