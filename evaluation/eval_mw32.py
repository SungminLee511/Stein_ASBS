"""
evaluation/eval_mw32.py

Evaluates ManyWell32D experiments (ASBS vs KSD-ASBS).

Metrics:
  - Mode coverage: count distinct sign patterns across 16 a-dimensions (max 2^16 = 65536)
  - Per-pair marginal W1: 1D Wasserstein on each of the 16 a-dimensions
  - Energy W2: 1D Wasserstein-2 between energy distributions
  - Mean energy: average energy of generated samples
  - Weight TV: how uniformly mass is split left/right across a-dimensions
  - Log-likelihood bound (ELBO proxy): mean log p(x) = -E(x) for generated samples

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python -u evaluation/eval_mw32.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Config
# ====================================================================

DEVICE = 'cuda'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
N_PAIRS = 16  # 16 independent 2D pairs
DIM = 32

EXPERIMENTS = {
    'ASBS': 'results/manywell32_asbs/seed_0',
    'KSD-ASBS': 'results/manywell32_ksd_asbs/seed_0',
}

OUTPUT_DIR = Path('evaluation/mw32_eval')


# ====================================================================
# Loading
# ====================================================================

def load_model(exp_dir, device):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / '.hydra' / 'config.yaml'

    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
    if not ckpt_path.exists():
        # Fall back to highest numbered checkpoint
        ckpts = sorted(exp_dir.glob('checkpoints/checkpoint_*.pt'),
                       key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else -1)
        ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint in {exp_dir / 'checkpoints'}")

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


# ====================================================================
# Sampling
# ====================================================================

@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


# ====================================================================
# Metrics
# ====================================================================

def get_well_positions():
    """Get left/right well positions from 1D double-well: a^4 - 6a^2 - 0.5a"""
    a = torch.linspace(-3, 3, 1000)
    e = a ** 4 - 6 * a ** 2 - 0.5 * a
    left_idx = e[:500].argmin()
    right_idx = e[500:].argmin() + 500
    return a[left_idx].item(), a[right_idx].item()


def evaluate_mode_coverage(samples, n_pairs=16):
    """Count distinct sign patterns in a-dimensions (left/right well assignment).
    Each a-dim has 2 wells. Sign pattern = binary vector of length 16.
    Max possible modes = 2^16 = 65536.
    """
    a_dims = samples[:, 0::2].cpu()  # (B, 16) — a-dimensions
    signs = (a_dims > 0).int()  # 1 = right well, 0 = left well

    # Convert to unique pattern IDs
    powers = 2 ** torch.arange(n_pairs)
    pattern_ids = (signs * powers).sum(dim=1)  # (B,)
    unique_patterns = pattern_ids.unique()

    # Per-dimension left/right fraction
    frac_right = (a_dims > 0).float().mean(dim=0)

    # Ideal: each dim has ~50% right (or ~85% right for asymmetric well)
    # Weight TV per dimension: deviation from reference fraction
    return {
        'unique_modes': len(unique_patterns),
        'total_modes': 2 ** n_pairs,
        'frac_right_per_dim': frac_right.tolist(),
        'mean_frac_right': frac_right.mean().item(),
    }


def evaluate_marginals(samples, ref_samples, n_pairs=16):
    """Per a-dimension W1 and per b-dimension W1."""
    results = {}
    a_w1_list = []
    b_w1_list = []

    for i in range(n_pairs):
        a_gen = samples[:, 2*i].cpu().numpy()
        a_ref = ref_samples[:, 2*i].cpu().numpy()
        b_gen = samples[:, 2*i+1].cpu().numpy()
        b_ref = ref_samples[:, 2*i+1].cpu().numpy()

        a_w1 = wasserstein_distance(a_gen, a_ref)
        b_w1 = wasserstein_distance(b_gen, b_ref)
        results[f'pair{i}_a_W1'] = a_w1
        results[f'pair{i}_b_W1'] = b_w1
        a_w1_list.append(a_w1)
        b_w1_list.append(b_w1)

    results['mean_a_W1'] = np.mean(a_w1_list)
    results['mean_b_W1'] = np.mean(b_w1_list)
    results['mean_all_W1'] = np.mean(a_w1_list + b_w1_list)
    return results


def evaluate_energy_w2(samples, ref_samples, energy):
    """1D Wasserstein-2 between energy distributions."""
    e_gen = energy.eval(samples).cpu().numpy().astype(np.float64)
    e_ref = energy.eval(ref_samples).cpu().numpy().astype(np.float64)

    # Filter NaN/Inf
    mask_g = np.isfinite(e_gen)
    mask_r = np.isfinite(e_ref)
    e_gen = e_gen[mask_g]
    e_ref = e_ref[mask_r]

    if len(e_gen) < 10 or len(e_ref) < 10:
        return float('nan'), float('nan'), float('nan')

    a = np.ones(len(e_gen)) / len(e_gen)
    b = np.ones(len(e_ref)) / len(e_ref)
    w2 = pot.emd2_1d(e_gen, e_ref, a, b) ** 0.5
    return float(w2), float(np.mean(e_gen)), float(np.mean(e_ref))


# ====================================================================
# Visualization
# ====================================================================

def plot_marginals(all_samples, ref_samples, output_path, n_show=8):
    """Show first n_show a-dimension marginals."""
    fig, axes = plt.subplots(2, n_show // 2, figsize=(4 * n_show // 2, 8))
    axes = axes.flatten()
    ref_np = ref_samples.cpu().numpy()

    colors = {'ASBS': '#d62728', 'KSD-ASBS': '#ff7f0e'}

    for idx in range(n_show):
        ax = axes[idx]
        dim = 2 * idx  # a-dimension of pair idx
        bins = np.linspace(-3.5, 3.5, 60)
        ax.hist(ref_np[:, dim], bins=bins, density=True, alpha=0.3, color='gray', label='Reference')
        for name, samples in all_samples.items():
            s = samples[:, dim].cpu().numpy()
            ax.hist(s, bins=bins, density=True, alpha=0.5, color=colors.get(name, 'blue'), label=name)
        ax.set_title(f'Pair {idx} ($a_{{{idx}}}$)')
        if idx == 0:
            ax.legend(fontsize=7)
        ax.set_xlim(-3.5, 3.5)

    fig.suptitle('ManyWell32D: Per-Pair a-Dimension Marginals (first 8 pairs)', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved marginal plot to {output_path}")


def plot_energy_hist(all_energies, ref_energy, output_path):
    """Energy distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'ASBS': '#d62728', 'KSD-ASBS': '#ff7f0e'}

    bins = np.linspace(ref_energy.min() - 10, ref_energy.max() + 10, 80)
    ax.hist(ref_energy, bins=bins, density=True, alpha=0.3, color='gray', label='Reference')
    for name, eng in all_energies.items():
        ax.hist(eng, bins=bins, density=True, alpha=0.5, color=colors.get(name, 'blue'), label=name)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
    ax.set_title('ManyWell32D: Energy Distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved energy histogram to {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    kst = timezone(timedelta(hours=9))
    print(f"ManyWell32D Evaluation — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_samples_for_plot = {}
    all_energies_for_plot = {}
    ref_samples = None
    ref_energy_np = None

    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {name}  ({exp_dir})")
        print(f"{'=' * 70}")

        print("  Loading model...")
        sde, source, energy, ts_cfg, epoch = load_model(exp_dir, DEVICE)
        print(f"  Loaded checkpoint from epoch {epoch}")

        # Generate reference samples (once)
        if ref_samples is None:
            print("  Generating reference samples...")
            ref_samples = energy.get_ref_samples(n=10000)
            if isinstance(ref_samples, torch.Tensor):
                ref_samples = ref_samples.to(DEVICE)
            else:
                ref_samples = torch.tensor(ref_samples, dtype=torch.float32).to(DEVICE)
            print(f"  Reference samples: {ref_samples.shape}")
            ref_energy_np = energy.eval(ref_samples).cpu().numpy()
            print(f"  Reference mean energy: {ref_energy_np.mean():.4f}")

        # Multi-seed evaluation
        seed_results = []
        all_seed_samples = []

        for eval_seed in range(N_EVAL_SEEDS):
            print(f"\n  --- Eval seed {eval_seed} ---")
            samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE, seed=eval_seed + 1000)

            # Check for NaN/Inf
            valid = ~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))
            n_valid = valid.sum().item()
            print(f"  Generated {N_SAMPLES} samples, {n_valid} valid")
            if n_valid < 100:
                print(f"  WARNING: Too few valid samples, skipping seed {eval_seed}")
                seed_results.append({'valid': n_valid, 'error': 'too_few_valid'})
                continue

            valid_samples = samples[valid]
            all_seed_samples.append(valid_samples)

            # Mode coverage
            mode_res = evaluate_mode_coverage(valid_samples)
            print(f"  Unique sign patterns: {mode_res['unique_modes']}/{mode_res['total_modes']}")
            print(f"  Mean frac_right: {mode_res['mean_frac_right']:.4f}")

            # Marginals
            marg_res = evaluate_marginals(valid_samples, ref_samples)
            print(f"  Mean a-dim W1: {marg_res['mean_a_W1']:.4f}")
            print(f"  Mean b-dim W1: {marg_res['mean_b_W1']:.4f}")
            print(f"  Mean all-dim W1: {marg_res['mean_all_W1']:.4f}")

            # Energy W2
            ew2, mean_E_gen, mean_E_ref = evaluate_energy_w2(valid_samples, ref_samples, energy)
            print(f"  Energy W2: {ew2:.4f}")
            print(f"  Mean energy (gen): {mean_E_gen:.4f}  (ref): {mean_E_ref:.4f}")

            seed_results.append({
                'valid': n_valid,
                'unique_modes': mode_res['unique_modes'],
                'mean_frac_right': mode_res['mean_frac_right'],
                'mean_a_W1': marg_res['mean_a_W1'],
                'mean_b_W1': marg_res['mean_b_W1'],
                'mean_all_W1': marg_res['mean_all_W1'],
                'energy_w2': ew2,
                'mean_energy': mean_E_gen,
            })

        # Aggregate across seeds
        valid_results = [r for r in seed_results if 'error' not in r]
        if valid_results:
            print(f"\n  {'=' * 50}")
            print(f"  AGGREGATE ({len(valid_results)} seeds) for {name}:")
            for key in ['unique_modes', 'mean_frac_right', 'mean_a_W1', 'mean_b_W1', 'energy_w2', 'mean_energy']:
                vals = [r[key] for r in valid_results]
                mean = np.mean(vals)
                std = np.std(vals)
                print(f"    {key}: {mean:.4f} ± {std:.4f}")
            print(f"  {'=' * 50}")

        all_results[name] = seed_results
        if all_seed_samples:
            all_samples_for_plot[name] = all_seed_samples[0]
            all_energies_for_plot[name] = energy.eval(all_seed_samples[0]).cpu().numpy()

    # Plots
    if all_samples_for_plot:
        print("\nGenerating marginal plot...")
        plot_marginals(all_samples_for_plot, ref_samples, OUTPUT_DIR / 'mw32_marginals.png')

        print("Generating energy histogram...")
        plot_energy_hist(all_energies_for_plot, ref_energy_np, OUTPUT_DIR / 'mw32_energy_hist.png')

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<20}"
    for name in EXPERIMENTS:
        header += f" | {name:>25}"
    print(header)
    print("-" * len(header))

    for key in ['unique_modes', 'mean_frac_right', 'mean_a_W1', 'mean_b_W1', 'energy_w2', 'mean_energy']:
        row = f"{key:<20}"
        for name in EXPERIMENTS:
            valid_results = [r for r in all_results[name] if 'error' not in r]
            if valid_results:
                vals = [r[key] for r in valid_results]
                row += f" | {np.mean(vals):>10.4f} ± {np.std(vals):.4f}"
            else:
                row += f" | {'FAILED':>25}"
        print(row)

    # Save results JSON
    json_path = OUTPUT_DIR / 'mw32_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {json_path}")

    print(f"\nDone! — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")


if __name__ == '__main__':
    main()
