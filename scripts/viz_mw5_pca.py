"""
scripts/viz_mw5_pca.py

PCA projection (5D → 2D) of MW5 samples to visualize mode coverage.
Produces a 3-panel figure: Reference | ASBS | KSD-ASBS
with mode centers projected alongside samples.

Usage:
  python scripts/viz_mw5_pca.py
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from omegaconf import OmegaConf
import hydra

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Config
# ====================================================================

DEVICE = 'cuda'
N_SAMPLES = 4000  # more samples for denser plots

EXPERIMENTS = {
    'ASBS (ckpt 4600)': {
        'dir': 'results/mw5_asbs/seed_0',
        'ckpt': 'results/mw5_asbs/seed_0/checkpoints/checkpoint_4600.pt',
    },
    'KSD-ASBS ($\\lambda$=0.5)': {
        'dir': 'results/mw5_ksd_asbs/seed_0',
        'ckpt': None,  # uses checkpoint_latest
    },
}

OUTPUT_DIR = Path('results/mw5_eval')


# ====================================================================
# Loading (reused from eval_mw5.py)
# ====================================================================

def load_model(exp_dir, device, ckpt_override=None):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / '.hydra' / 'config.yaml'

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
    epoch = ckpt.get('epoch', '?')

    ts_cfg = {
        't0': float(cfg.timesteps.t0),
        't1': float(cfg.timesteps.t1),
        'steps': int(cfg.timesteps.steps),
        'rescale_t': cfg.timesteps.rescale_t if cfg.timesteps.rescale_t is not None else None,
    }
    return sde, source, energy, ts_cfg, epoch


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device, seed=42):
    torch.manual_seed(seed)
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


# ====================================================================
# Visualization
# ====================================================================

def plot_mw5_pca(all_samples, ref_samples, mode_centers, output_path):
    """3-panel PCA projection: Reference | ASBS | KSD-ASBS.

    PCA is fit on reference samples, then all data is projected consistently.
    Mode centers are shown as red stars.
    """
    # Fit PCA on reference
    ref_np = ref_samples.cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(ref_np)

    ref_2d = pca.transform(ref_np)
    centers_2d = pca.transform(mode_centers.cpu().numpy())

    # Project all experiment samples
    projected = {}
    for name, samples in all_samples.items():
        projected[name] = pca.transform(samples.cpu().numpy())

    n_panels = 1 + len(projected)  # Reference + experiments
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

    # Color settings
    sample_alpha = 0.15
    sample_size = 3
    center_size = 120
    center_color = '#d62728'

    # Panel 0: Reference
    ax = axes[0]
    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], s=sample_size, alpha=sample_alpha,
               c='#1f77b4', rasterized=True)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], s=center_size, c=center_color,
               marker='*', edgecolors='black', linewidths=0.5, zorder=10, label='Mode centers')
    ax.set_title('Reference (10k samples)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # Experiment panels
    exp_colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    for i, (name, pts_2d) in enumerate(projected.items()):
        ax = axes[i + 1]
        color = exp_colors[i % len(exp_colors)]
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1], s=sample_size, alpha=sample_alpha,
                   c=color, rasterized=True)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], s=center_size, c=center_color,
                   marker='*', edgecolors='black', linewidths=0.5, zorder=10)

        # Count covered modes
        samples_t = torch.tensor(pts_2d, dtype=torch.float32)
        centers_t = torch.tensor(centers_2d, dtype=torch.float32)
        # Use original 5D for mode counting (more accurate)
        orig_samples = list(all_samples.values())[i]
        orig_centers = mode_centers.to(orig_samples.device)
        dists = torch.cdist(orig_samples, orig_centers)
        assignments = dists.argmin(dim=1)
        min_dists = dists.min(dim=1).values
        covered = 0
        for k in range(mode_centers.shape[0]):
            mask = (assignments == k)
            if mask.any() and min_dists[mask].min() < 1.0:
                covered += 1

        ax.set_title(f'{name}\n({covered}/32 modes covered)', fontsize=13, fontweight='bold')

    # Shared formatting
    # Compute axis limits from reference
    pad = 0.15
    x_min, x_max = ref_2d[:, 0].min(), ref_2d[:, 0].max()
    y_min, y_max = ref_2d[:, 1].min(), ref_2d[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    for ax in axes:
        ax.set_xlim(x_min - pad * x_range, x_max + pad * x_range)
        ax.set_ylim(y_min - pad * y_range, y_max + pad * y_range)
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.2)

    fig.suptitle('MW5: PCA Projection (5D $\\to$ 2D) — Mode Coverage Comparison',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PCA figure to {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    kst = timezone(timedelta(hours=9))
    print(f"MW5 PCA Visualization — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = {}
    ref_samples = None
    mode_centers = None
    energy = None

    for name, exp_cfg in EXPERIMENTS.items():
        print(f"\nLoading: {name}")
        sde, source, eng, ts_cfg, epoch = load_model(
            exp_cfg['dir'], DEVICE, ckpt_override=exp_cfg.get('ckpt'))
        print(f"  Checkpoint epoch: {epoch}")

        if energy is None:
            energy = eng

        if ref_samples is None:
            print("  Generating reference samples...")
            ref_samples = energy.get_ref_samples(n=10000)
            if isinstance(ref_samples, torch.Tensor):
                ref_samples = ref_samples.to(DEVICE)
            else:
                ref_samples = torch.tensor(ref_samples, dtype=torch.float32).to(DEVICE)
            mode_centers = energy.get_mode_centers().to(DEVICE)
            print(f"  Reference: {ref_samples.shape}, Mode centers: {mode_centers.shape}")

        print("  Generating samples...")
        samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE, seed=42)
        print(f"  Generated {samples.shape[0]} samples")
        all_samples[name] = samples

    # Plot
    print("\nGenerating PCA figure...")
    plot_mw5_pca(all_samples, ref_samples, mode_centers,
                 OUTPUT_DIR / 'mw5_pca_mode_coverage.png')

    print(f"\nDone! — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")


if __name__ == '__main__':
    main()
