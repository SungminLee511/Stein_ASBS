"""
Plot marginal evolution for Grid25 ASBS seed 5 (diverged run).
Uses checkpoint_800.pt — last valid checkpoint before divergence at ~epoch 867.
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
FIG_DIR = Path('/home/TRASH_CAN/output_port')
XLIM = (-6, 6)
YLIM = (-6, 6)


def load_model(exp_dir, device, ckpt_path):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / 'config.yaml'
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


def plot_marginal_evolution(states, ts, centers, method_name, output_path, n_snapshots=5):
    T = len(states)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5.5))
    color = '#2ca02c'  # green for seed 5

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

    fig.suptitle(f'Grid25 ASBS Seed 5 — Marginal Evolution (ckpt 800, diverged @ ~867)',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    exp_dir = RESULTS_DIR / 'grid25_asbs' / 'seed_5'
    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_800.pt'

    print(f"Loading seed 5 from {ckpt_path} ...")
    sde, source, energy, ts_cfg = load_model(exp_dir, device, ckpt_path)
    centers = energy.get_centers().to(device)

    n_samples = 2000
    print(f"Generating {n_samples} full trajectories...")
    torch.manual_seed(42)
    states, ts = generate_full_states(sde, source, ts_cfg, n_samples, device)

    output_path = FIG_DIR / 'grid25_seed5_marginal_ckpt800.png'
    print("Plotting marginal evolution...")
    plot_marginal_evolution(states, ts, centers, 'ASBS seed5', output_path)

    print("Done!")


if __name__ == '__main__':
    main()
