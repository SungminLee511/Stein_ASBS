"""
scripts/eval_mw5_pca_marginal.py

MW5 PCA (5D -> 2D) marginal evolution for all 9 experiments.
Each row = one experiment, each column = one timestep snapshot.
PCA is fit on reference samples so all panels share the same projection.
Mode centers (32) are plotted as markers.
"""

import sys
sys.path.insert(0, '/home/sky/SML/Stein_ASBS')

import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra

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

C_ASBS = '#c0392b'
C_KSD1 = '#e67e22'
C_KSD5 = '#2980b9'
C_REF = '#555555'
C_BG = '#fafafa'

TEXT_WIDTH_IN = 5.5

EXPERIMENTS = [
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
def generate_full_states(sde, source, ts_cfg, n_samples, device):
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    states = sdeint(sde, x0, ts, only_boundary=False)
    return states, ts


# ====================================================================
# PCA Marginal Evolution Plot
# ====================================================================

def plot_mw5_pca_marginal(all_states, all_ts, all_labels, all_colors,
                           pca, centers_2d, ref_2d, output_path, n_snapshots=5):
    """Multi-row PCA scatter marginal evolution."""
    n_rows = len(all_states)
    n_cols = n_snapshots

    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95
    fig_w = TEXT_WIDTH_IN + 0.7
    fig_h = row_h * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Compute axis limits from reference
    pad = 0.5
    xlim = (ref_2d[:, 0].min() - pad, ref_2d[:, 0].max() + pad)
    ylim = (ref_2d[:, 1].min() - pad, ref_2d[:, 1].max() + pad)

    for row, (states, ts, label, color) in enumerate(
        zip(all_states, all_ts, all_labels, all_colors)
    ):
        T = len(states)
        indices = np.linspace(0, T - 1, n_cols, dtype=int)
        for col, si in enumerate(indices):
            ax = axes[row, col]
            t_val = ts[si].item()
            s_5d = states[si].cpu().numpy()
            s_2d = pca.transform(s_5d)

            ax.set_facecolor(C_BG)

            # Mode centers
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='+', s=20,
                       c='black', linewidths=0.5, zorder=10)

            # Samples
            ax.scatter(s_2d[:, 0], s_2d[:, 1], s=0.8, c=color, alpha=0.25,
                       zorder=5, edgecolors='none', rasterized=True)

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect('equal')

            if row == 0:
                ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)

            ax.tick_params(length=2, width=0.4, labelsize=5)
            ax.set_xticks([]); ax.set_yticks([])

            if col == 0 and row == n_rows - 1:
                ax.set_xlabel('PC1', fontsize=7)
                ax.set_ylabel('PC2', fontsize=7)

        axes[row, 0].set_ylabel(label, fontsize=6.5, fontweight='bold', labelpad=8)

    fig.subplots_adjust(wspace=0.06, hspace=0.10)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_mw5_pca_terminal(pca, centers_2d, ref_2d, all_terminal_2d,
                           all_labels, all_colors, output_path):
    """Terminal PCA scatter: GT + all experiments."""
    n_panels = 1 + len(all_terminal_2d)
    n_cols = min(5, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    col_w = TEXT_WIDTH_IN / n_cols
    row_h = col_w * 0.95
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(TEXT_WIDTH_IN, row_h * n_rows))
    axes_flat = np.array(axes).flatten()

    pad = 0.5
    xlim = (ref_2d[:, 0].min() - pad, ref_2d[:, 0].max() + pad)
    ylim = (ref_2d[:, 1].min() - pad, ref_2d[:, 1].max() + pad)

    # GT
    ax = axes_flat[0]
    ax.set_facecolor(C_BG)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='+', s=20, c='black',
               linewidths=0.5, zorder=10)
    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], s=0.8, c=C_REF, alpha=0.25,
               zorder=5, edgecolors='none', rasterized=True)
    ax.set_title('Reference', fontsize=7, pad=4)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

    for i, (s_2d, label, color) in enumerate(
        zip(all_terminal_2d, all_labels, all_colors)
    ):
        ax = axes_flat[i + 1]
        ax.set_facecolor(C_BG)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='+', s=20, c='black',
                   linewidths=0.5, zorder=10)
        ax.scatter(s_2d[:, 0], s_2d[:, 1], s=0.8, c=color, alpha=0.25,
                   zorder=5, edgecolors='none', rasterized=True)
        ax.set_title(label, fontsize=6.5, pad=4)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(wspace=0.08, hspace=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    set_neurips_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 2000
    print("=" * 60)
    print("  MW5 PCA Marginal Evolution (9 experiments)")
    print("=" * 60)

    # Load first experiment to get energy/ref
    first_dir = RESULTS_DIR / EXPERIMENTS[0][0]
    cfg_path = first_dir / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = first_dir / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)
    energy = hydra.utils.instantiate(cfg.energy, device=DEVICE)

    # Reference samples and mode centers
    ref = energy.get_ref_samples(5000)
    ref_samples = (ref if isinstance(ref, torch.Tensor)
                   else torch.tensor(ref, dtype=torch.float32))
    ref_np = ref_samples.cpu().numpy()

    centers = energy.get_mode_centers().cpu().numpy()  # (32, 5)

    # Fit PCA on reference samples
    print("  Fitting PCA on reference samples...")
    pca = PCA(n_components=2)
    pca.fit(ref_np)
    ref_2d = pca.transform(ref_np)
    centers_2d = pca.transform(centers)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

    all_states = []
    all_ts = []
    all_terminal_2d = []
    all_labels = []
    all_colors = []

    for exp_rel, label, color in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_rel
        print(f"\n  Loading {label} from {exp_dir}...")
        try:
            sde, source, _, ts_cfg = load_model(exp_dir, DEVICE)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        torch.manual_seed(42)
        states, ts = generate_full_states(sde, source, ts_cfg, n_samples, DEVICE)

        # Check divergence
        terminal = states[-1]
        E = energy.eval(terminal.to(DEVICE))
        e_mean = E.mean().item()
        if np.isnan(e_mean) or np.isinf(e_mean) or abs(e_mean) > 1e10:
            print(f"    SKIP: diverged (E_mean={e_mean:.2e})")
            del sde, source, states
            torch.cuda.empty_cache()
            continue

        all_states.append(states)
        all_ts.append(ts)
        terminal_2d = pca.transform(terminal.cpu().numpy())
        all_terminal_2d.append(terminal_2d)
        all_labels.append(label)
        all_colors.append(color)
        print(f"    OK (E_mean={e_mean:.2f})")

        del sde, source
        torch.cuda.empty_cache()

    # ---- PCA Marginal Evolution ----
    print(f"\n  Generating PCA marginal evolution ({len(all_states)} rows)...")
    plot_mw5_pca_marginal(
        all_states, all_ts, all_labels, all_colors,
        pca, centers_2d, ref_2d,
        FIG_DIR / 'mw5_pca_marginal_neurips.png',
    )

    # ---- PCA Terminal ----
    print("  Generating PCA terminal distribution...")
    plot_mw5_pca_terminal(
        pca, centers_2d, ref_2d, all_terminal_2d,
        all_labels, all_colors,
        FIG_DIR / 'mw5_pca_terminal_neurips.png',
    )

    print("\n  Done!")


if __name__ == '__main__':
    main()
