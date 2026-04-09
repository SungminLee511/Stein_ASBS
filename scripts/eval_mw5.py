"""
scripts/eval_mw5.py

Evaluates MW5 experiments (ASBS vs KSD-ASBS) following the evaluation plan:
  - Mode coverage (32 modes)
  - Per-dimension marginal W1
  - Energy W2
  - Per-dimension marginal histograms (5-panel figure)

Usage:
  python scripts/eval_mw5.py
"""

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

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Config
# ====================================================================

DEVICE = 'cuda'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5  # seeds 0-4, 2000 samples each

EXPERIMENTS = {
    'ASBS': 'results/mw5_asbs/seed_0',
    'KSD-ASBS (λ=0.5)': 'results/mw5_ksd_asbs/seed_0',
}

OUTPUT_DIR = Path('results/mw5_eval')


# ====================================================================
# Loading
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

def evaluate_mw5_modes(samples, energy):
    """Count how many of 32 modes are covered."""
    centers = energy.get_mode_centers().to(samples.device)  # (32, 5)
    N = samples.shape[0]

    dists = torch.cdist(samples, centers)  # (N, 32)
    assignments = dists.argmin(dim=1)  # (N,)
    min_dists = dists.min(dim=1).values  # (N,)

    threshold = 1.0
    covered_mask = torch.zeros(32, dtype=torch.bool)
    for k in range(32):
        mode_samples = (assignments == k)
        if mode_samples.any():
            if min_dists[mode_samples].min() < threshold:
                covered_mask[k] = True

    counts = torch.zeros(32)
    for k in range(32):
        counts[k] = (assignments == k).sum()

    results = {
        'modes_covered': covered_mask.sum().item(),
        'modes_total': 32,
        'mode_counts': counts.cpu().tolist(),
        'weight_TV': 0.5 * (counts / N - 1.0 / 32).abs().sum().item(),
        'energy_mean': energy.eval(samples).mean().item(),
    }
    return results


def evaluate_mw5_marginals(samples, ref_samples):
    """Per-dimension W1 and bimodality check."""
    results = {}
    for d in range(5):
        gen = samples[:, d].cpu().numpy()
        ref = ref_samples[:, d].cpu().numpy()
        w1 = wasserstein_distance(gen, ref)
        results[f'dim{d}_W1'] = w1
        frac_left_gen = (gen < 0).mean()
        frac_left_ref = (ref < 0).mean()
        results[f'dim{d}_frac_left_gen'] = float(frac_left_gen)
        results[f'dim{d}_frac_left_ref'] = float(frac_left_ref)
    results['mean_W1'] = np.mean([results[f'dim{d}_W1'] for d in range(5)])
    return results


def evaluate_energy_w2(samples, ref_samples, energy):
    """1D Wasserstein-2 between energy distributions."""
    import ot
    e_gen = energy.eval(samples).cpu().numpy().astype(np.float64)
    e_ref = energy.eval(ref_samples).cpu().numpy().astype(np.float64)
    a = np.ones(len(e_gen)) / len(e_gen)
    b = np.ones(len(e_ref)) / len(e_ref)
    w2 = ot.emd2_1d(e_gen, e_ref, a, b) ** 0.5
    return float(w2)


# ====================================================================
# Visualization
# ====================================================================

def plot_mw5_marginals(all_samples, ref_samples, output_path):
    """5-panel figure: per-dimension marginal histograms."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    ref_np = ref_samples.cpu().numpy()

    colors = {'ASBS': '#d62728', 'KSD-ASBS (λ=0.5)': '#ff7f0e'}

    for d in range(5):
        ax = axes[d]
        bins = np.linspace(-3.5, 3.5, 60)
        ax.hist(ref_np[:, d], bins=bins, density=True, alpha=0.3, color='gray', label='Reference')
        for name, samples in all_samples.items():
            s = samples[:, d].cpu().numpy()
            ax.hist(s, bins=bins, density=True, alpha=0.5, color=colors.get(name, 'blue'), label=name)
        ax.set_title(f'$x_{d+1}$')
        if d == 0:
            ax.legend(fontsize=8)
        ax.set_xlim(-3.5, 3.5)

    fig.suptitle('MW5: Per-Dimension Marginals', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved marginal plot to {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    kst = timezone(timedelta(hours=9))
    print(f"MW5 Evaluation — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Store results and samples for plotting
    all_results = {}
    all_samples_for_plot = {}
    ref_samples = None

    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {name}  ({exp_dir})")
        print(f"{'=' * 70}")

        # Load model
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

        # Multi-seed evaluation
        seed_results = []
        all_seed_samples = []

        for eval_seed in range(N_EVAL_SEEDS):
            print(f"\n  --- Eval seed {eval_seed} ---")
            samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE, seed=eval_seed + 1000)
            print(f"  Generated {samples.shape[0]} samples")
            all_seed_samples.append(samples)

            # Mode coverage
            mode_res = evaluate_mw5_modes(samples, energy)
            print(f"  Modes covered: {mode_res['modes_covered']}/{mode_res['modes_total']}")
            print(f"  Weight TV: {mode_res['weight_TV']:.4f}")
            print(f"  Mean energy: {mode_res['energy_mean']:.4f}")

            # Marginals
            marg_res = evaluate_mw5_marginals(samples, ref_samples)
            print(f"  Mean marginal W1: {marg_res['mean_W1']:.4f}")
            for d in range(5):
                print(f"    dim{d}: W1={marg_res[f'dim{d}_W1']:.4f}  "
                      f"frac_left={marg_res[f'dim{d}_frac_left_gen']:.3f} "
                      f"(ref={marg_res[f'dim{d}_frac_left_ref']:.3f})")

            # Energy W2
            ew2 = evaluate_energy_w2(samples, ref_samples, energy)
            print(f"  Energy W2: {ew2:.4f}")

            seed_results.append({
                'modes_covered': mode_res['modes_covered'],
                'weight_TV': mode_res['weight_TV'],
                'energy_mean': mode_res['energy_mean'],
                'mean_W1': marg_res['mean_W1'],
                'energy_w2': ew2,
            })

        # Aggregate across seeds
        print(f"\n  {'=' * 50}")
        print(f"  AGGREGATE ({N_EVAL_SEEDS} seeds) for {name}:")
        for key in ['modes_covered', 'weight_TV', 'energy_mean', 'mean_W1', 'energy_w2']:
            vals = [r[key] for r in seed_results]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"    {key}: {mean:.4f} ± {std:.4f}")
        print(f"  {'=' * 50}")

        all_results[name] = seed_results
        # Use seed 0 samples for plotting
        all_samples_for_plot[name] = all_seed_samples[0]

    # Plot marginals
    print("\nGenerating marginal plot...")
    plot_mw5_marginals(all_samples_for_plot, ref_samples, OUTPUT_DIR / 'mw5_marginals.png')

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<20}"
    for name in EXPERIMENTS:
        header += f" | {name:>20}"
    print(header)
    print("-" * len(header))

    for key in ['modes_covered', 'weight_TV', 'energy_mean', 'mean_W1', 'energy_w2']:
        row = f"{key:<20}"
        for name in EXPERIMENTS:
            vals = [r[key] for r in all_results[name]]
            row += f" | {np.mean(vals):>8.4f} ± {np.std(vals):.4f}"
        print(row)

    print(f"\nDone! — {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}")


if __name__ == '__main__':
    main()
