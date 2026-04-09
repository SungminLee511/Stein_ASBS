"""
scripts/eval_unequal_gmm.py

Evaluates unequal-weight GMM (5-mode) benchmark:
  - 3-panel terminal distribution: Ground Truth | ASBS | KSD-ASBS
  - 5-panel marginal evolution per method
  - Mode weight bar chart (KEY figure)
  - Appends results to evaluation/2d_result.md
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
    """Sample from the mixture by ancestral sampling."""
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
    """The key metric: does the model recover the correct mode weights?"""
    centers = energy.get_centers().to(samples.device)
    true_weights = energy.get_weights().to(samples.device)
    std = energy.get_std()
    N = samples.shape[0]

    # Assign each sample to nearest mode
    dists = torch.cdist(samples, centers)
    assignments = dists.argmin(dim=1)

    # Empirical weights
    empirical_weights = torch.zeros(5, device=samples.device)
    for k in range(5):
        empirical_weights[k] = (assignments == k).float().mean()

    # Per-mode counts
    counts = torch.zeros(5, dtype=torch.long)
    for k in range(5):
        counts[k] = (assignments == k).sum()

    # Modes covered (threshold-based)
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


def plot_terminal(energy, samples_ref, samples_base, samples_ksd,
                  centers, m_base, m_ksd, output_path, xlim, ylim):
    """3-panel: Ground Truth | ASBS | KSD-ASBS"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    titles = [
        'Ground Truth',
        f'ASBS\n({m_base["n_modes_covered"]}/{m_base["n_modes_total"]} modes)',
        f'KSD-ASBS (λ=1.0)\n({m_ksd["n_modes_covered"]}/{m_ksd["n_modes_total"]} modes)',
    ]
    sample_sets = [samples_ref, samples_base, samples_ksd]
    colors = ['gray', '#d62728', '#ff7f0e']

    for ax, title, samples, color in zip(axes, titles, sample_sets, colors):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_facecolor('#f7f7f7')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=100, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=4, c=color, alpha=0.4, zorder=5)

    fig.suptitle('Unequal-Weight GMM (5 modes): Terminal Distribution', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_marginal_evolution(energy, states, ts, centers, std,
                            method_name, output_path, xlim, ylim, n_snapshots=5):
    """5-panel marginal evolution"""
    T = len(states)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5.5))
    color = '#d62728' if method_name == 'ASBS' else '#ff7f0e'

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


def plot_weight_bar_chart(results_base, results_ksd, output_path):
    """Bar chart: true weights vs baseline vs KSD-ASBS per mode. KEY FIGURE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(5)
    width = 0.25

    true_w = results_base['true_weights']
    base_w = results_base['empirical_weights']
    ksd_w = results_ksd['empirical_weights']

    bars1 = ax.bar(x - width, true_w, width, label='Target', color='#4c72b0', alpha=0.8)
    bars2 = ax.bar(x, base_w, width, label='ASBS (Baseline)', color='#d62728', alpha=0.8)
    bars3 = ax.bar(x + width, ksd_w, width, label='KSD-ASBS (λ=1.0)', color='#ff7f0e', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Mode {i+1}\n(w={true_w[i]:.2f})' for i in range(5)], fontsize=11)
    ax.set_ylabel('Empirical Weight', fontsize=13)
    ax.set_title('Mode Weight Recovery: Unequal-Weight GMM', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    # Annotate minority mode
    ax.axhline(y=0.03, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(4.5, 0.035, 'Target: 3%', fontsize=9, color='gray', ha='right')

    ax.set_ylim(0, max(max(true_w), max(base_w), max(ksd_w)) * 1.15)

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
    print(f"=== Unequal-Weight GMM Evaluation ===")
    print(f"Time: {kst}")

    baseline_dir = RESULTS_DIR / 'unequal_gmm_asbs' / 'seed_0'
    ksd_dir = RESULTS_DIR / 'unequal_gmm_ksd_asbs' / 'seed_0'

    xlim = (-8, 8)
    ylim = (-8, 8)

    # Load models
    print("Loading baseline ASBS...")
    sde_base, src_base, energy, ts_base = load_model(baseline_dir, device)
    print("Loading KSD-ASBS (lambda=1.0)...")
    sde_ksd, src_ksd, _, ts_ksd = load_model(ksd_dir, device)

    centers = energy.get_centers().to(device)

    # --- Generate terminal samples ---
    n_samples = 2000
    print(f"Generating {n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_ksd = generate_samples(sde_ksd, src_ksd, ts_ksd, n_samples, device)
    torch.manual_seed(123)
    samples_ref = generate_reference_samples(energy, n_samples).to(device)

    # --- Metrics ---
    print("Computing metrics...")
    m_base = evaluate_unequal_gmm(samples_base, energy)
    m_ksd = evaluate_unequal_gmm(samples_ksd, energy)

    print(f"\n{'='*50}")
    print(f"  ASBS Baseline:")
    print(f"    Modes covered: {m_base['n_modes_covered']}/5")
    print(f"    Empirical weights: {[f'{w:.4f}' for w in m_base['empirical_weights']]}")
    print(f"    True weights:      {m_base['true_weights']}")
    print(f"    Weight TV: {m_base['weight_TV']:.4f}")
    print(f"    Weight KL: {m_base['weight_KL']:.4f}")
    print(f"    Minority mode (3%): weight={m_base['minority_mode_weight']:.4f}, alive={m_base['minority_mode_alive']}")
    print(f"    Mean energy: {m_base['mean_energy']:.4f}")
    print(f"    Per-mode counts: {m_base['per_mode_counts']}")

    print(f"\n  KSD-ASBS (lambda=1.0):")
    print(f"    Modes covered: {m_ksd['n_modes_covered']}/5")
    print(f"    Empirical weights: {[f'{w:.4f}' for w in m_ksd['empirical_weights']]}")
    print(f"    True weights:      {m_ksd['true_weights']}")
    print(f"    Weight TV: {m_ksd['weight_TV']:.4f}")
    print(f"    Weight KL: {m_ksd['weight_KL']:.4f}")
    print(f"    Minority mode (3%): weight={m_ksd['minority_mode_weight']:.4f}, alive={m_ksd['minority_mode_alive']}")
    print(f"    Mean energy: {m_ksd['mean_energy']:.4f}")
    print(f"    Per-mode counts: {m_ksd['per_mode_counts']}")
    print(f"{'='*50}\n")

    # --- Figures ---
    print("Generating figures...")

    # 1. Terminal distribution (3-panel)
    plot_terminal(
        energy, samples_ref, samples_base, samples_ksd,
        centers, m_base, m_ksd,
        FIG_DIR / 'unequal_gmm_terminal.png', xlim, ylim,
    )

    # 2. Mode weight bar chart (KEY figure)
    plot_weight_bar_chart(
        m_base, m_ksd,
        FIG_DIR / 'unequal_gmm_weights.png',
    )

    # 3. Marginal evolution
    print("Generating marginal evolution figures...")
    torch.manual_seed(42)
    states_base, ts_full_base = generate_full_states(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(42)
    states_ksd, ts_full_ksd = generate_full_states(sde_ksd, src_ksd, ts_ksd, n_samples, device)

    plot_marginal_evolution(
        energy, states_base, ts_full_base, centers, energy.get_std(),
        'ASBS', FIG_DIR / 'unequal_gmm_marginal_asbs.png', xlim, ylim,
    )
    plot_marginal_evolution(
        energy, states_ksd, ts_full_ksd, centers, energy.get_std(),
        'KSD-ASBS', FIG_DIR / 'unequal_gmm_marginal_ksd.png', xlim, ylim,
    )

    # --- Append to 2d_result.md ---
    print("Appending results to 2d_result.md...")
    append_to_results_md(kst, m_base, m_ksd)

    print("\n=== Done! ===")


def append_to_results_md(kst, m_base, m_ksd):
    md_path = EVAL_DIR / '2d_result.md'

    section = f"""
## Unequal-Weight GMM (5 modes, unequal weights)

Evaluated: {kst}

Target weights: [0.50, 0.25, 0.15, 0.07, 0.03]

| Metric | ASBS (Baseline) | KSD-ASBS (λ=1.0) |
|---|---|---|
| Modes covered (of 5) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} |
| Mode weight TV ↓ | {m_base['weight_TV']:.4f} | {m_ksd['weight_TV']:.4f} |
| Weight KL ↓ | {m_base['weight_KL']:.4f} | {m_ksd['weight_KL']:.4f} |
| Minority mode weight (target: 0.03) | {m_base['minority_mode_weight']:.4f} | {m_ksd['minority_mode_weight']:.4f} |
| Minority mode alive? | {m_base['minority_mode_alive']} | {m_ksd['minority_mode_alive']} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} |
| Std energy | {m_base['std_energy']:.4f} | {m_ksd['std_energy']:.4f} |
| Empirical weights | {[f'{w:.4f}' for w in m_base['empirical_weights']]} | {[f'{w:.4f}' for w in m_ksd['empirical_weights']]} |
| Per-mode counts | {m_base['per_mode_counts']} | {m_ksd['per_mode_counts']} |

### Mode Weight Recovery

![unequal_gmm weights](figures_2d/unequal_gmm_weights.png)

### Terminal Distribution

![unequal_gmm terminal](figures_2d/unequal_gmm_terminal.png)

### Marginal Evolution: ASBS

![unequal_gmm marginal asbs](figures_2d/unequal_gmm_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![unequal_gmm marginal ksd](figures_2d/unequal_gmm_marginal_ksd.png)

---

"""
    with open(md_path, 'a') as f:
        f.write(section)
    print(f"  Appended to {md_path}")


if __name__ == '__main__':
    main()
