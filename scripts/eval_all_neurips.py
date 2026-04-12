"""
scripts/eval_all_neurips.py

Comprehensive NeurIPS evaluation for all Grid25 and MW5 experiments.
- Grid25: ASBS (s0-2), KSD λ=1 (s1-2), KSD λ=5 (s0-2)
- MW5:    ASBS (s0-2), KSD λ=1 (s0-2), KSD λ=5 (s0-2)

Generates:
  1. Metrics table (mode coverage, W1, energy W2, Sinkhorn, weight TV)
  2. Grid25 marginal evolution (NeurIPS 2D scatter, rows = experiments)
  3. MW5 marginal evolution (per-dim histograms at 5 timesteps, rows = experiments)
  4. Terminal distribution comparison panels
"""

import sys
sys.path.insert(0, '/home/sky/SML/Stein_ASBS')

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
import ot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# NeurIPS style
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
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# ====================================================================
# Constants
# ====================================================================

RESULTS_DIR = Path('/home/sky/SML/Stein_ASBS/results')
FIG_DIR = Path('/home/sky/SML/Stein_ASBS/evaluation/figures_2d')
DEVICE = 'cuda'

C_REF = '#555555'
C_ASBS = '#c0392b'       # deep red
C_KSD1 = '#e67e22'       # warm orange
C_KSD5 = '#2980b9'       # blue
C_CONTOUR = '#2c3e50'
C_BG = '#fafafa'

TEXT_WIDTH_IN = 5.5

GRID25_EXPERIMENTS = [
    ('grid25_asbs/seed_0',         'ASBS s0',       C_ASBS),
    ('grid25_asbs/seed_1',         'ASBS s1',       C_ASBS),
    ('grid25_asbs/seed_2',         'ASBS s2',       C_ASBS),
    ('grid25_ksd_asbs/seed_1',     'KSD λ=1 s1',   C_KSD1),
    ('grid25_ksd_asbs/seed_2',     'KSD λ=1 s2',   C_KSD1),
    ('grid25_ksd_asbs_lam5/seed_0','KSD λ=5 s0',   C_KSD5),
    ('grid25_ksd_asbs_lam5/seed_1','KSD λ=5 s1',   C_KSD5),
    ('grid25_ksd_asbs_lam5/seed_2','KSD λ=5 s2',   C_KSD5),
]

MW5_EXPERIMENTS = [
    ('mw5_asbs/seed_0',           'ASBS s0',       C_ASBS),
    ('mw5_asbs/seed_1',           'ASBS s1',       C_ASBS),
    ('mw5_asbs/seed_2',           'ASBS s2',       C_ASBS),
    ('mw5_ksd_asbs/seed_0',       'KSD λ=1 s0',   C_KSD1),
    ('mw5_ksd_asbs/seed_1',       'KSD λ=1 s1',   C_KSD1),
    ('mw5_ksd_asbs/seed_2',       'KSD λ=1 s2',   C_KSD1),
    ('mw5_ksd_asbs_lam5/seed_0',  'KSD λ=5 s0',   C_KSD5),
    ('mw5_ksd_asbs_lam5/seed_1',  'KSD λ=5 s1',   C_KSD5),
    ('mw5_ksd_asbs_lam5/seed_2',  'KSD λ=5 s2',   C_KSD5),
]


# ====================================================================
# Loading & sampling
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


# ====================================================================
# Metrics
# ====================================================================

def assign_modes_2d(samples, centers, threshold_factor=3.0, std=0.3):
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
    return (counts > 0).sum().item(), counts


def evaluate_mw5_modes(samples, energy):
    centers = energy.get_mode_centers().to(samples.device)
    dists = torch.cdist(samples, centers)
    assignments = dists.argmin(dim=1)
    min_dists = dists.min(dim=1).values
    threshold = 1.0
    covered = torch.zeros(32, dtype=torch.bool)
    for k in range(32):
        mask = (assignments == k)
        if mask.any() and min_dists[mask].min() < threshold:
            covered[k] = True
    counts = torch.zeros(32)
    for k in range(32):
        counts[k] = (assignments == k).sum()
    N = samples.shape[0]
    return covered.sum().item(), 0.5 * (counts / N - 1.0 / 32).abs().sum().item()


def compute_metrics(samples, ref_samples, energy, is_grid25=True):
    """Compute comprehensive metrics."""
    N = samples.shape[0]
    metrics = {}

    # Mode coverage
    if is_grid25:
        centers = energy.get_centers().to(samples.device)
        std = energy.get_std()
        n_covered, counts = assign_modes_2d(samples, centers, std=std)
        metrics['modes'] = f"{n_covered}/25"
        metrics['n_modes'] = n_covered
        metrics['weight_TV'] = 0.5 * (counts.float() / N - 1.0 / 25).abs().sum().item()
    else:
        n_covered, weight_tv = evaluate_mw5_modes(samples, energy)
        metrics['modes'] = f"{n_covered}/32"
        metrics['n_modes'] = n_covered
        metrics['weight_TV'] = weight_tv

    # Energy stats
    E = energy.eval(samples)
    metrics['E_mean'] = E.mean().item()
    metrics['E_std'] = E.std().item()

    # W2
    a = samples.cpu().numpy().astype(np.float64)
    b = ref_samples.cpu().numpy().astype(np.float64)
    wa = np.ones(len(a)) / len(a)
    wb = np.ones(len(b)) / len(b)
    M = ot.dist(a, b, metric='sqeuclidean')
    w2 = float(np.sqrt(ot.emd2(wa, wb, M)))
    metrics['W2'] = w2

    # Sinkhorn
    sinkhorn = float(ot.sinkhorn2(wa, wb, M, reg=0.1))
    metrics['Sinkhorn'] = sinkhorn

    # Per-dim W1 (for MW5)
    if not is_grid25:
        w1s = []
        for d in range(samples.shape[1]):
            w1 = wasserstein_distance(
                samples[:, d].cpu().numpy(),
                ref_samples[:, d].cpu().numpy()
            )
            w1s.append(w1)
        metrics['mean_W1'] = np.mean(w1s)

    return metrics


# ====================================================================
# Grid25 Visualization
# ====================================================================

def plot_grid25_marginal_big(all_states, all_ts, all_labels, all_colors,
                              energy, output_path, n_snapshots=5):
    """Multi-row × 5-col: each row is one experiment's marginal evolution."""
    n_rows = len(all_states)
    n_cols = n_snapshots
    centers = energy.get_centers()
    c_np = centers.cpu().numpy()

    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95
    fig_w = TEXT_WIDTH_IN + 0.7
    fig_h = row_h * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    XLIM = (-6, 6)
    YLIM = (-6, 6)

    for row, (states, ts, label, color) in enumerate(
        zip(all_states, all_ts, all_labels, all_colors)
    ):
        T = len(states)
        indices = np.linspace(0, T - 1, n_cols, dtype=int)
        for col, si in enumerate(indices):
            ax = axes[row, col]
            t_val = ts[si].item()
            s = states[si].cpu().numpy()
            ax.set_facecolor(C_BG)
            ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=10, c='black',
                       linewidths=0.3, zorder=10)
            ax.scatter(s[:, 0], s[:, 1], s=1.0, c=color, alpha=0.3,
                       zorder=5, edgecolors='none', rasterized=True)
            ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)
            ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
            ax.tick_params(length=2, width=0.4, labelsize=6)
            if col > 0:
                ax.set_yticklabels([])
            if row < n_rows - 1:
                ax.set_xticklabels([])
        axes[row, 0].set_ylabel(label, fontsize=7, fontweight='bold', labelpad=8)

    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_grid25_terminal(energy, ref_samples, all_terminal, all_labels,
                          all_colors, all_modes, output_path):
    """Terminal distribution panels."""
    n_panels = 1 + len(all_terminal)
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH_IN, row_h * n_rows))
    axes_flat = np.array(axes).flatten()
    centers = energy.get_centers()
    c_np = centers.cpu().numpy()
    XLIM = (-6, 6); YLIM = (-6, 6)

    # GT panel
    ax = axes_flat[0]
    ax.set_facecolor(C_BG)
    ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=15, c='black', linewidths=0.5, zorder=10)
    s = ref_samples.cpu().numpy()
    ax.scatter(s[:, 0], s[:, 1], s=1.5, c=C_REF, alpha=0.35, zorder=5,
               edgecolors='none', rasterized=True)
    ax.set_title('Ground Truth', fontsize=8, pad=4)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
    ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
    ax.tick_params(length=2, width=0.4)

    for i, (samples, label, color, modes) in enumerate(
        zip(all_terminal, all_labels, all_colors, all_modes)
    ):
        ax = axes_flat[i + 1]
        ax.set_facecolor(C_BG)
        ax.scatter(c_np[:, 0], c_np[:, 1], marker='+', s=15, c='black', linewidths=0.5, zorder=10)
        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=1.5, c=color, alpha=0.35, zorder=5,
                   edgecolors='none', rasterized=True)
        ax.set_title(f'{label} ({modes})', fontsize=7, pad=4)
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        ax.tick_params(length=2, width=0.4)

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(wspace=0.2, hspace=0.35)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# MW5 Visualization
# ====================================================================

def plot_mw5_marginal_evolution(all_states, all_ts, all_labels, all_colors,
                                 ref_samples, output_path, n_snapshots=5):
    """MW5 marginal evolution: rows = experiments, cols = timesteps.
    Each panel shows per-dim histograms overlaid."""
    n_rows = len(all_states)
    n_cols = n_snapshots

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH_IN + 0.7, 1.3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    ref_np = ref_samples.cpu().numpy()
    bins = np.linspace(-4, 4, 50)

    for row, (states, ts, label, color) in enumerate(
        zip(all_states, all_ts, all_labels, all_colors)
    ):
        T = len(states)
        indices = np.linspace(0, T - 1, n_cols, dtype=int)
        for col, si in enumerate(indices):
            ax = axes[row, col]
            t_val = ts[si].item()
            s = states[si].cpu().numpy()

            # Plot all 5 dims overlaid as histograms
            for d in range(5):
                ax.hist(s[:, d], bins=bins, density=True, alpha=0.15, color=color,
                        linewidth=0)
            # Overlay reference at terminal panel
            if col == n_cols - 1:
                for d in range(5):
                    ax.hist(ref_np[:, d], bins=bins, density=True, alpha=0.1,
                            color='gray', linewidth=0)

            ax.set_xlim(-4, 4)
            ax.set_ylim(0, 1.0)
            if row == 0:
                ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)
            ax.tick_params(length=2, width=0.4, labelsize=5)
            if col > 0:
                ax.set_yticklabels([])
            if row < n_rows - 1:
                ax.set_xticklabels([])
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([0, 0.5])

        axes[row, 0].set_ylabel(label, fontsize=6.5, fontweight='bold', labelpad=6)

    fig.subplots_adjust(wspace=0.08, hspace=0.15)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mw5_terminal_marginals(all_terminal, all_labels, all_colors,
                                 ref_samples, output_path):
    """MW5 terminal: 5-panel per-dimension marginal comparison."""
    n_methods = len(all_terminal)
    fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH_IN, 1.8))
    ref_np = ref_samples.cpu().numpy()
    bins = np.linspace(-4, 4, 60)

    for d in range(5):
        ax = axes[d]
        ax.hist(ref_np[:, d], bins=bins, density=True, alpha=0.3, color='gray',
                label='Ref' if d == 0 else None)
        for samples, label, color in zip(all_terminal, all_labels, all_colors):
            s = samples[:, d].cpu().numpy()
            ax.hist(s, bins=bins, density=True, alpha=0.25, color=color,
                    label=label if d == 0 else None)
        ax.set_title(f'$x_{d+1}$', fontsize=8)
        ax.set_xlim(-4, 4)
        ax.tick_params(length=2, width=0.4, labelsize=6)
        ax.set_xticks([-2, 0, 2])

    axes[0].legend(fontsize=5, loc='upper left', ncol=1)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def run_benchmark(name, experiments, is_grid25, n_samples=2000):
    """Run eval + viz for one benchmark."""
    print(f"\n{'#' * 70}")
    print(f"  {name}")
    print(f"{'#' * 70}")

    energy = None
    ref_samples = None

    all_states = []
    all_ts = []
    all_terminal = []
    all_labels = []
    all_colors = []
    all_modes = []
    all_metrics = []
    valid_indices = []

    for i, (exp_rel, label, color) in enumerate(experiments):
        exp_dir = RESULTS_DIR / exp_rel
        print(f"\n  [{label}] Loading from {exp_dir}...")
        try:
            sde, source, en, ts_cfg = load_model(exp_dir, DEVICE)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        if energy is None:
            energy = en
            if is_grid25:
                ref_samples = energy.get_ref_samples(5000).to(DEVICE)
            else:
                ref = energy.get_ref_samples(5000)
                ref_samples = (ref if isinstance(ref, torch.Tensor)
                               else torch.tensor(ref, dtype=torch.float32)).to(DEVICE)

        # Generate full trajectory
        torch.manual_seed(42)
        states, ts = generate_full_states(sde, source, ts_cfg, n_samples, DEVICE)
        terminal = states[-1]

        # Check if diverged (loss > 1000 means bad terminal samples)
        E = energy.eval(terminal)
        e_mean = E.mean().item()
        if np.isnan(e_mean) or np.isinf(e_mean) or abs(e_mean) > 1e10:
            print(f"    SKIP: diverged (E_mean={e_mean:.2e})")
            del sde, source, states
            torch.cuda.empty_cache()
            continue

        # Metrics
        metrics = compute_metrics(terminal, ref_samples, energy, is_grid25)
        metrics['label'] = label
        all_metrics.append(metrics)

        all_states.append(states)
        all_ts.append(ts)
        all_terminal.append(terminal)
        all_labels.append(label)
        all_colors.append(color)
        all_modes.append(metrics['modes'])
        valid_indices.append(i)

        print(f"    Modes: {metrics['modes']} | W2: {metrics['W2']:.4f} | "
              f"Sinkhorn: {metrics['Sinkhorn']:.4f} | TV: {metrics['weight_TV']:.4f}")
        if 'mean_W1' in metrics:
            print(f"    mean_W1: {metrics['mean_W1']:.4f}")

        del sde, source
        torch.cuda.empty_cache()

    if not all_states:
        print(f"  No valid experiments for {name}!")
        return all_metrics

    # ---- Metrics table ----
    print(f"\n  {'=' * 70}")
    print(f"  {name} — METRICS SUMMARY")
    print(f"  {'=' * 70}")
    header_keys = ['modes', 'W2', 'Sinkhorn', 'weight_TV', 'E_mean']
    if not is_grid25:
        header_keys.append('mean_W1')
    header = f"  {'Experiment':<18}"
    for k in header_keys:
        header += f" {k:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in all_metrics:
        row = f"  {m['label']:<18}"
        for k in header_keys:
            v = m[k]
            if isinstance(v, str):
                row += f" {v:>12}"
            else:
                row += f" {v:>12.4f}"
        print(row)
    print(f"  {'=' * 70}")

    # ---- Visualizations ----
    prefix = 'grid25' if is_grid25 else 'mw5'

    if is_grid25:
        print(f"\n  Generating Grid25 marginal evolution...")
        plot_grid25_marginal_big(
            all_states, all_ts, all_labels, all_colors,
            energy, FIG_DIR / f'{prefix}_all_marginal_neurips.png',
        )
        print(f"  Generating Grid25 terminal distribution...")
        plot_grid25_terminal(
            energy, ref_samples[:n_samples], all_terminal, all_labels,
            all_colors, all_modes, FIG_DIR / f'{prefix}_all_terminal_neurips.png',
        )
    else:
        print(f"\n  Generating MW5 marginal evolution...")
        plot_mw5_marginal_evolution(
            all_states, all_ts, all_labels, all_colors,
            ref_samples, FIG_DIR / f'{prefix}_all_marginal_neurips.png',
        )
        print(f"  Generating MW5 terminal marginals...")
        plot_mw5_terminal_marginals(
            all_terminal, all_labels, all_colors,
            ref_samples, FIG_DIR / f'{prefix}_all_terminal_neurips.png',
        )

    return all_metrics


def main():
    set_neurips_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kst = timezone(timedelta(hours=9))
    print(f"Full Evaluation — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 70)

    # Grid25
    grid25_metrics = run_benchmark("Grid25 (5×5 GMM)", GRID25_EXPERIMENTS, is_grid25=True)

    # MW5
    mw5_metrics = run_benchmark("MW5 (5D Many-Well)", MW5_EXPERIMENTS, is_grid25=False)

    # Save metrics JSON
    results = {
        'grid25': grid25_metrics,
        'mw5': mw5_metrics,
        'timestamp': datetime.now(kst).isoformat(),
    }
    json_path = FIG_DIR / 'eval_all_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics saved to {json_path}")

    print(f"\nDone! — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")


if __name__ == '__main__':
    main()
