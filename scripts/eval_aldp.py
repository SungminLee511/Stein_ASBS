"""
Evaluate alanine dipeptide: compute Ramachandran plot (phi, psi) for
baseline and SDR-ASBS, compare mode coverage in dihedral space.

Usage:
    python scripts/eval_aldp.py \
        --ref_path data/test_split_ALDP-10000.npy \
        --asbs_ckpt results/aldp_asbs/seed_0/checkpoints/checkpoint_latest.pt \
        --ksd_ckpt results/aldp_ksd_asbs/seed_0/checkpoints/checkpoint_latest.pt \
        --as_ckpt results/aldp_as/seed_0/checkpoints/checkpoint_latest.pt \
        --output_dir results/aldp_eval
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adjoint_samplers.energies.aldp_energy import AlanineDipeptideEnergy


def compute_basin_occupancy(phi_deg, psi_deg):
    """Count samples in known Ramachandran basins."""
    n = len(phi_deg)
    basins = {}

    # C7eq: phi ~ -80, psi ~ 80 (dominant basin)
    mask = (phi_deg > -120) & (phi_deg < -40) & (psi_deg > 40) & (psi_deg < 120)
    basins['C7eq'] = mask.sum()

    # C7ax: phi ~ 75, psi ~ -65
    mask = (phi_deg > 40) & (phi_deg < 120) & (psi_deg > -110) & (psi_deg < -20)
    basins['C7ax'] = mask.sum()

    # alphaR: phi ~ -80, psi ~ -40
    mask = (phi_deg > -120) & (phi_deg < -40) & (psi_deg > -80) & (psi_deg < 0)
    basins['alphaR'] = mask.sum()

    # alphaL: phi ~ 60, psi ~ 40
    mask = (phi_deg > 20) & (phi_deg < 100) & (psi_deg > 0) & (psi_deg < 80)
    basins['alphaL'] = mask.sum()

    # C5: phi ~ -155, psi ~ 155
    mask = (phi_deg > -180) & (phi_deg < -120) & (psi_deg > 120) & (psi_deg < 180)
    basins['C5'] = mask.sum()

    occupied = sum(1 for v in basins.values() if v > 0)
    return basins, occupied


def load_and_generate_samples(ckpt_path, energy, n_samples=2000, device='cpu'):
    """Load a checkpoint and generate samples."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    ckpt = torch.load(ckpt_path, map_location=device)

    # Try to load config from checkpoint directory
    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, 'cfg.yaml')

    if not os.path.exists(cfg_path):
        print(f"  Warning: cfg.yaml not found at {cfg_path}")
        return None

    cfg = OmegaConf.load(cfg_path)

    # Reconstruct model and generate samples
    # This depends on the exact training infrastructure
    # For now, return None and we'll implement after seeing training output
    print(f"  Loaded checkpoint from {ckpt_path}")
    return None


def plot_ramachandran(datasets, labels, colors, output_path, title="Ramachandran Plot"):
    """Plot Ramachandran diagrams for multiple datasets."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (phi, psi), label, color in zip(axes, datasets, labels, colors):
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)

        ax.scatter(phi_deg, psi_deg, s=1, alpha=0.3, c=color)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\psi$ (degrees)', fontsize=12)
        ax.set_title(label, fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add basin labels
        ax.annotate('C7eq', xy=(-80, 80), fontsize=8, color='gray',
                     ha='center', va='center')
        ax.annotate('C7ax', xy=(75, -65), fontsize=8, color='gray',
                     ha='center', va='center')
        ax.annotate(r'$\alpha_R$', xy=(-80, -40), fontsize=8, color='gray',
                     ha='center', va='center')
        ax.annotate(r'$\alpha_L$', xy=(60, 40), fontsize=8, color='gray',
                     ha='center', va='center')

        # Basin stats
        basins, occupied = compute_basin_occupancy(phi_deg, psi_deg)
        stats_text = f"Basins: {occupied}/5\n"
        for k, v in basins.items():
            if v > 0:
                stats_text += f"{k}: {v} ({100*v/len(phi_deg):.1f}%)\n"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Ramachandran plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str,
                        default='data/test_split_ALDP-10000.npy')
    parser.add_argument('--asbs_ckpt', type=str, default=None)
    parser.add_argument('--ksd_ckpt', type=str, default=None)
    parser.add_argument('--as_ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/aldp_eval')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize energy (needed for dihedral computation)
    energy = AlanineDipeptideEnergy(dim=66, temperature=300.0, platform='CPU')

    # Load reference samples
    ref_samples = np.load(args.ref_path)
    print(f"Reference samples: {ref_samples.shape}")

    ref_tensor = torch.tensor(ref_samples, dtype=torch.float32)
    ref_phi, ref_psi = energy.compute_dihedrals(ref_tensor)
    ref_phi_np = ref_phi.numpy()
    ref_psi_np = ref_psi.numpy()

    # Reference-only Ramachandran plot
    datasets = [(ref_phi_np, ref_psi_np)]
    labels = ['Reference (MD)']
    colors = ['steelblue']

    plot_ramachandran(datasets, labels, colors,
                      os.path.join(args.output_dir, 'ramachandran_reference.png'),
                      title='Reference Ramachandran')

    # Basin stats for reference
    ref_phi_deg = np.degrees(ref_phi_np)
    ref_psi_deg = np.degrees(ref_psi_np)
    basins, occupied = compute_basin_occupancy(ref_phi_deg, ref_psi_deg)
    print(f"\nReference basin occupancy ({occupied}/5 basins):")
    for k, v in basins.items():
        print(f"  {k}: {v} ({100*v/len(ref_phi_deg):.1f}%)")

    # Save metrics
    metrics = {
        'reference': {
            'n_samples': len(ref_samples),
            'basins': {k: int(v) for k, v in basins.items()},
            'occupied_basins': occupied,
        }
    }

    with open(os.path.join(args.output_dir, 'aldp_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {args.output_dir}/aldp_metrics.json")
    print("Done!")


if __name__ == '__main__':
    main()
