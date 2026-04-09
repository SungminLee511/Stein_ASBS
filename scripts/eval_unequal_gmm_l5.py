"""
scripts/eval_unequal_gmm_l5.py

Evaluates unequal-weight GMM (5-mode) with 3 methods:
  - ASBS (Baseline)
  - KSD-ASBS (λ=1.0)
  - KSD-ASBS (λ=5.0)

Generates:
  - 4-panel terminal distribution: Ground Truth | ASBS | KSD λ=1 | KSD λ=5
  - 5-panel marginal evolution per method (3 sets)
  - 4-group bar chart for mode weight recovery
  - Prints comparison table for 2d_result.md
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
# Sample generation
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


def generate_reference_samples(energy, n=2000):
    weights = energy.get_weights()
    centers = energy.get_centers()
    std = energy.get_std()
    mode_idx = torch.multinomial(weights, n, replacement=True)
    samples = centers[mode_idx] + std * torch.randn(n, 2)
    return samples


# ====================================================================
# Metrics
# ====================================================================

def evaluate_unequal_gmm(samples, energy):
    centers = energy.get_centers().to(samples.device)
    true_weights = energy.get_weights().to(samples.device)
    std = energy.get_std()
    N = samples.shape[0]

    dists = torch.cdist(samples, centers)
    assignments = dists.argmin(dim=1)

    empirical_weights = torch.zeros(5, device=samples.device)
    for k in range(5):
        empirical_weights[k] = (assignments == k).float().mean()

    counts = torch.zeros(5, dtype=torch.long)
    for k in range(5):
        counts[k] = (assignments == k).sum()

    min_dists = dists.min(dim=1).values
    threshold = 3.0 * std
    covered_mask = torch.zeros(5, dtype=torch.bool)
    for k in range(5):
        mode_samples = (assignments == k)
        if mode_samples.any():
            if min_dists[mode_samples].min() < threshold:
                covered_mask[k] = True
    n_covered = covered_mask.sum().item()

    E = energy.eval(samples)

    results = {
        'empirical_weights': empirical_weights.cpu().tolist(),
        'true_weights': true_weights.cpu().tolist(),
        'weight_TV': 0.5 * (empirical_weights - true_weights).abs().sum().item(),
        'minority_mode_alive': bool(empirical_weights[4] > 0),
        'minority_mode_weight': empirical_weights[4].item(),
        'weight_KL': (true_weights * torch.log(true_weights / (empirical_weights + 1e-10))).sum().item(),
        'n_modes_covered': n_covered,
        'n_modes_total': 5,
        'per_mode_counts': counts.cpu().tolist(),
        'mean_energy': E.mean().item(),
        'std_energy': E.std().item(),
    }
    return results


# ====================================================================
# Plotting
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
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(), levels=10, colors='steelblue', alpha=0.4, linewidths=0.5)


def plot_terminal_4panel(energy, samples_ref, samples_base, samples_ksd1, samples_ksd5,
                          centers, m_base, m_ksd1, m_ksd5, output_path, xlim, ylim):
    """4-panel: Ground Truth | ASBS | KSD λ=1 | KSD λ=5"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

    titles = [
        'Ground Truth',
        f'ASBS\n({m_base["n_modes_covered"]}/{m_base["n_modes_total"]} modes)',
        f'KSD-ASBS (λ=1.0)\n({m_ksd1["n_modes_covered"]}/{m_ksd1["n_modes_total"]} modes)',
        f'KSD-ASBS (λ=5.0)\n({m_ksd5["n_modes_covered"]}/{m_ksd5["n_modes_total"]} modes)',
    ]
    sample_sets = [samples_ref, samples_base, samples_ksd1, samples_ksd5]
    colors = ['gray', '#d62728', '#ff7f0e', '#2ca02c']

    for ax, title, samples, color in zip(axes, titles, sample_sets, colors):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_facecolor('#f7f7f7')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=100, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=4, c=color, alpha=0.4, zorder=5)

    fig.suptitle('Unequal-Weight GMM (5 modes): Terminal Distribution Comparison', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_marginal_evolution(energy, states, ts, centers, std,
                            method_name, color, output_path, xlim, ylim, n_snapshots=5):
    """5-panel marginal evolution"""
    T = len(states)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5.5))

    for panel_idx, state_idx in enumerate(indices):
        ax = axes[panel_idx]
        t_val = ts[state_idx].item()
        samples = states[state_idx].cpu().numpy()

        ax.scatter(samples[:, 0], samples[:, 1], s=4, c=color, alpha=0.4, zorder=5)
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=80, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_facecolor('#f7f7f7')
        ax.set_title(f't = {t_val:.2f}', fontsize=13, fontweight='bold')

    fig.suptitle(f'Unequal-Weight GMM: {method_name} — Marginal Evolution', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_weight_bar_chart_3way(m_base, m_ksd1, m_ksd5, output_path):
    """Bar chart: true weights vs baseline vs KSD λ=1 vs KSD λ=5 per mode."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(5)
    width = 0.2

    true_w = m_base['true_weights']
    base_w = m_base['empirical_weights']
    ksd1_w = m_ksd1['empirical_weights']
    ksd5_w = m_ksd5['empirical_weights']

    bars1 = ax.bar(x - 1.5*width, true_w, width, label='Target', color='#4c72b0', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, base_w, width, label='ASBS (Baseline)', color='#d62728', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, ksd1_w, width, label='KSD-ASBS (λ=1.0)', color='#ff7f0e', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, ksd5_w, width, label='KSD-ASBS (λ=5.0)', color='#2ca02c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Mode {i+1}\n(w={true_w[i]:.2f})' for i in range(5)], fontsize=11)
    ax.set_ylabel('Empirical Weight', fontsize=13)
    ax.set_title('Mode Weight Recovery: Unequal-Weight GMM — 3-Way Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=0.03, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(4.5, 0.035, 'Target: 3%', fontsize=9, color='gray', ha='right')

    all_w = true_w + base_w + ksd1_w + ksd5_w
    ax.set_ylim(0, max(all_w) * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

RESULTS_DIR = Path('/home/RESEARCH/Stein_ASBS/results')
EVAL_DIR = Path('/home/RESEARCH/Stein_ASBS/evaluation')
FIG_DIR = EVAL_DIR / 'figures_2d'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"=== Unequal-Weight GMM — 3-Way Evaluation (Baseline / λ=1.0 / λ=5.0) ===")
    print(f"Time: {kst}")

    baseline_dir = RESULTS_DIR / 'unequal_gmm_asbs' / 'seed_0'
    ksd1_dir = RESULTS_DIR / 'unequal_gmm_ksd_asbs' / 'seed_0'
    ksd5_dir = RESULTS_DIR / 'unequal_gmm_ksd_asbs_l5' / 'seed_0'

    xlim = (-8, 8)
    ylim = (-8, 8)

    # Load models
    print("Loading baseline ASBS...")
    sde_base, src_base, energy, ts_base = load_model(baseline_dir, device)
    print("Loading KSD-ASBS (λ=1.0)...")
    sde_ksd1, src_ksd1, _, ts_ksd1 = load_model(ksd1_dir, device)
    print("Loading KSD-ASBS (λ=5.0)...")
    sde_ksd5, src_ksd5, _, ts_ksd5 = load_model(ksd5_dir, device)

    centers = energy.get_centers().to(device)

    # --- Generate terminal samples ---
    n_samples = 2000
    print(f"Generating {n_samples} terminal samples per method...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_ksd1 = generate_samples(sde_ksd1, src_ksd1, ts_ksd1, n_samples, device)
    torch.manual_seed(0)
    samples_ksd5 = generate_samples(sde_ksd5, src_ksd5, ts_ksd5, n_samples, device)
    torch.manual_seed(123)
    samples_ref = generate_reference_samples(energy, n_samples).to(device)

    # --- Metrics ---
    print("Computing metrics...")
    m_base = evaluate_unequal_gmm(samples_base, energy)
    m_ksd1 = evaluate_unequal_gmm(samples_ksd1, energy)
    m_ksd5 = evaluate_unequal_gmm(samples_ksd5, energy)

    for label, m in [("ASBS Baseline", m_base), ("KSD-ASBS (λ=1.0)", m_ksd1), ("KSD-ASBS (λ=5.0)", m_ksd5)]:
        print(f"\n  {label}:")
        print(f"    Modes covered: {m['n_modes_covered']}/5")
        print(f"    Empirical weights: {[f'{w:.4f}' for w in m['empirical_weights']]}")
        print(f"    True weights:      {m['true_weights']}")
        print(f"    Weight TV: {m['weight_TV']:.4f}")
        print(f"    Weight KL: {m['weight_KL']:.4f}")
        print(f"    Minority mode (3%): weight={m['minority_mode_weight']:.4f}, alive={m['minority_mode_alive']}")
        print(f"    Mean energy: {m['mean_energy']:.4f}")
        print(f"    Std energy: {m['std_energy']:.4f}")
        print(f"    Per-mode counts: {m['per_mode_counts']}")

    # --- Figures ---
    print("\nGenerating figures...")

    # 1. Terminal distribution (4-panel)
    plot_terminal_4panel(
        energy, samples_ref, samples_base, samples_ksd1, samples_ksd5,
        centers, m_base, m_ksd1, m_ksd5,
        FIG_DIR / 'unequal_gmm_l5_terminal.png', xlim, ylim,
    )

    # 2. Mode weight bar chart (KEY figure)
    plot_weight_bar_chart_3way(
        m_base, m_ksd1, m_ksd5,
        FIG_DIR / 'unequal_gmm_l5_weights.png',
    )

    # 3. Marginal evolution for λ=5.0 (the new one)
    print("Generating marginal evolution for KSD-ASBS λ=5.0...")
    torch.manual_seed(42)
    states_ksd5, ts_full_ksd5 = generate_full_states(sde_ksd5, src_ksd5, ts_ksd5, n_samples, device)
    plot_marginal_evolution(
        energy, states_ksd5, ts_full_ksd5, centers, energy.get_std(),
        'KSD-ASBS (λ=5.0)', '#2ca02c',
        FIG_DIR / 'unequal_gmm_marginal_ksd_l5.png', xlim, ylim,
    )

    print("\n=== Done! ===")


if __name__ == '__main__':
    main()
