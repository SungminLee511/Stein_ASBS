"""
scripts/eval_grid25_neurips.py

NeurIPS-quality evaluation of Grid25 benchmark.
Generates publication figures and comprehensive metrics.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from omegaconf import OmegaConf
import hydra
import ot  # POT library for Wasserstein / Sinkhorn

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# NeurIPS style
# ====================================================================

def set_neurips_style():
    """Configure matplotlib for NeurIPS publication quality."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        # Lines & markers
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        # Axes
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        # Ticks
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # Figure
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
    })


# ====================================================================
# Constants
# ====================================================================

RESULTS_DIR = Path('/home/RESEARCH/Stein_ASBS/results')
EVAL_DIR = Path('/home/RESEARCH/Stein_ASBS/evaluation')
FIG_DIR = EVAL_DIR / 'figures_2d'

XLIM = (-6, 6)
YLIM = (-6, 6)

# Colors
C_REF = '#555555'
C_ASBS = '#c0392b'      # deep red
C_SDR = '#e67e22'        # warm orange
C_CONTOUR = '#2c3e50'    # dark blue-gray for contours
C_BG = '#fafafa'         # very light gray background

TEXT_WIDTH_IN = 5.5  # NeurIPS single-column text width


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
# Sampling
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


# ====================================================================
# Metrics
# ====================================================================

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


def compute_mode_weight_tv(counts):
    """Total variation distance between empirical mode weights and uniform."""
    K = len(counts)
    total = sum(counts)
    if total == 0:
        return 1.0
    empirical = np.array(counts, dtype=np.float64) / total
    uniform = np.ones(K) / K
    return 0.5 * np.abs(empirical - uniform).sum()


def compute_kl_divergence(samples, energy, n_grid=300):
    """Estimate forward KL: KL(p_model || p_target) via density ratio on grid.

    Uses kernel density estimation for the model distribution
    and the known energy for the target.
    """
    from scipy.stats import gaussian_kde

    s = samples.cpu().numpy().T  # (2, N)
    try:
        kde = gaussian_kde(s, bw_method='silverman')
    except np.linalg.LinAlgError:
        return float('nan')

    # Evaluate on grid
    x = np.linspace(XLIM[0], XLIM[1], n_grid)
    y = np.linspace(YLIM[0], YLIM[1], n_grid)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=0)  # (2, n_grid^2)

    log_q = kde.logpdf(grid_pts)  # model log density

    grid_tensor = torch.tensor(grid_pts.T, dtype=torch.float32)
    with torch.no_grad():
        E = energy.eval(grid_tensor).cpu().numpy()  # (n_grid^2,)
    log_p_unnorm = -E
    # Normalize target on grid
    log_Z = np.log(np.exp(log_p_unnorm - log_p_unnorm.max()).sum()) + log_p_unnorm.max()
    dx = (x[1] - x[0]) * (y[1] - y[0])
    log_Z += np.log(dx)
    log_p = log_p_unnorm - log_Z

    # KL(q || p) = E_q[log q - log p]
    q_density = np.exp(log_q)
    valid = q_density > 1e-10
    kl = np.sum(q_density[valid] * (log_q[valid] - log_p[valid])) * dx
    return float(kl)


def compute_w2_distance(samples, ref_samples):
    """Wasserstein-2 distance using POT library."""
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    # Uniform weights
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    w2_sq = ot.emd2(wa, wb, M)
    return float(np.sqrt(w2_sq))


def compute_sinkhorn_divergence(samples, ref_samples, reg=0.1):
    """Sinkhorn divergence using POT library."""
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    sinkhorn = ot.sinkhorn2(wa, wb, M, reg=reg)
    return float(sinkhorn)


def compute_ess(log_weights):
    """Effective sample size from log importance weights."""
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    ess = torch.exp(-torch.logsumexp(2 * log_w, dim=0))
    return ess.item()


def compute_ess_from_samples(samples, energy, source):
    """ESS via importance weights: w(x) = p_target(x) / p_source(x).

    For the source log_prob, we use the known Gaussian N(0, scale^2 I).
    """
    with torch.no_grad():
        log_p_target = -energy.eval(samples)
        # Source is Gauss(dim, scale) — compute log_prob manually
        # (the custom Gauss class doesn't implement log_prob)
        scale = float(source.scale) if hasattr(source, 'scale') else 1.0
        loc = float(source.loc) if hasattr(source, 'loc') else 0.0
        d = samples.shape[-1]
        log_p_source = (
            -0.5 * d * np.log(2 * np.pi)
            - d * np.log(scale)
            - 0.5 * ((samples - loc) ** 2).sum(dim=-1) / (scale ** 2)
        )
    log_w = log_p_target - log_p_source
    return compute_ess(log_w)


def compute_all_metrics(samples, energy, centers, std, ref_samples, source):
    """Compute all metrics."""
    assignments, counts, n_covered = assign_modes(
        samples, centers.to(samples.device), std=std
    )
    E = energy.eval(samples)
    counts_list = counts.cpu().tolist()

    metrics = {
        'n_modes_covered': n_covered,
        'n_modes_total': centers.shape[0],
        'per_mode_counts': counts_list,
        'mean_energy': E.mean().item(),
        'std_energy': E.std().item(),
        'mode_weight_tv': compute_mode_weight_tv(counts_list),
        'w2': compute_w2_distance(samples, ref_samples),
        'sinkhorn': compute_sinkhorn_divergence(samples, ref_samples),
    }

    print("    Computing KL divergence (grid-based)...")
    metrics['kl'] = compute_kl_divergence(samples, energy)

    print("    Computing ESS...")
    try:
        metrics['ess'] = compute_ess_from_samples(samples, energy, source)
        metrics['ess_pct'] = metrics['ess'] / len(samples) * 100
    except Exception as e:
        print(f"    ESS failed: {e}")
        metrics['ess'] = float('nan')
        metrics['ess_pct'] = float('nan')

    return metrics


# ====================================================================
# Plotting helpers
# ====================================================================

def plot_density_contours(ax, energy, xlim, ylim, n_grid=250):
    """Plot target density contours in background."""
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


# ====================================================================
# Figure 1: Terminal Distribution (3-panel)
# ====================================================================

def plot_terminal_neurips(
    energy, samples_ref, samples_base, samples_ksd,
    centers, m_base, m_ksd, output_path,
):
    fig_h = TEXT_WIDTH_IN / 3 * 0.95  # roughly square panels
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, fig_h))

    titles = [
        r'Ground Truth',
        f'ASBS ({m_base["n_modes_covered"]}/{m_base["n_modes_total"]})',
        f'SDR-ASBS ({m_ksd["n_modes_covered"]}/{m_ksd["n_modes_total"]})',
    ]
    sample_sets = [samples_ref, samples_base, samples_ksd]
    colors = [C_REF, C_ASBS, C_SDR]

    for ax, title, samples, color in zip(axes, titles, sample_sets, colors):
        ax.set_facecolor(C_BG)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')

        # Density contours
        plot_density_contours(ax, energy, XLIM, YLIM)

        # Mode center markers
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='+', s=15, c='black',
                   linewidths=0.5, zorder=10)

        # Samples
        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=1.5, c=color, alpha=0.35, zorder=5,
                   edgecolors='none', rasterized=True)

        # Clean ticks
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(length=2, width=0.4)

    axes[0].set_ylabel(r'$x_2$', fontsize=9)
    for ax in axes:
        ax.set_xlabel(r'$x_1$', fontsize=9)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path} + .pdf")


# ====================================================================
# Figure 2: Marginal Evolution (5-panel per method)
# ====================================================================

def plot_marginal_neurips(
    energy, states, ts, centers, std,
    method_name, output_path, color,
    n_snapshots=5,
):
    T = len(states)
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

    fig_h = TEXT_WIDTH_IN / n_snapshots * 0.95
    fig, axes = plt.subplots(1, n_snapshots, figsize=(TEXT_WIDTH_IN, fig_h))

    for panel_idx, state_idx in enumerate(indices):
        ax = axes[panel_idx]
        t_val = ts[state_idx].item()
        samples = states[state_idx].cpu().numpy()

        ax.set_facecolor(C_BG)

        # Mode center markers
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='+', s=12, c='black',
                   linewidths=0.4, zorder=10)

        # Samples
        ax.scatter(samples[:, 0], samples[:, 1], s=1.5, c=color, alpha=0.35,
                   zorder=5, edgecolors='none', rasterized=True)

        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.tick_params(length=2, width=0.4, labelsize=7)

        if panel_idx == 0:
            ax.set_ylabel(r'$x_2$', fontsize=8)

    fig.subplots_adjust(wspace=0.25)
    # Method name as left-side text to avoid overlap with panel titles
    fig.text(0.01, 0.5, method_name, fontsize=10, fontweight='bold',
             va='center', ha='left', rotation=90)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
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

    baseline_dir = RESULTS_DIR / 'grid25_asbs' / 'seed_0'
    ksd_dir = RESULTS_DIR / 'grid25_ksd_asbs_lam01' / 'seed_0'
    ksd_lambda = 0.1
    n_samples = 2000

    print("=" * 60)
    print("  Grid25 — NeurIPS Evaluation")
    print("=" * 60)

    # Load
    print("  Loading ASBS...")
    sde_base, src_base, energy, ts_base = load_model(baseline_dir, device)
    print("  Loading SDR-ASBS...")
    sde_ksd, src_ksd, _, ts_ksd = load_model(ksd_dir, device)

    centers = energy.get_centers().to(device)
    std = energy.get_std()

    # Generate samples
    print(f"  Generating {n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_ksd = generate_samples(sde_ksd, src_ksd, ts_ksd, n_samples, device)

    # Reference samples (more for accurate W2)
    ref_samples = energy.get_ref_samples(5000).to(device)
    ref_for_plot = ref_samples[:n_samples]

    # ---- Metrics ----
    print("\n  === ASBS Metrics ===")
    m_base = compute_all_metrics(samples_base, energy, centers, std, ref_samples, src_base)
    print("\n  === SDR-ASBS Metrics ===")
    m_ksd = compute_all_metrics(samples_ksd, energy, centers, std, ref_samples, src_ksd)

    # Print results table
    print("\n" + "=" * 70)
    print(f"  {'Metric':<30} {'ASBS':>15} {'SDR-ASBS':>15}")
    print("-" * 70)
    print(f"  {'Modes covered (of 25)':<30} {m_base['n_modes_covered']:>15} {m_ksd['n_modes_covered']:>15}")
    print(f"  {'Mean energy':<30} {m_base['mean_energy']:>15.4f} {m_ksd['mean_energy']:>15.4f}")
    print(f"  {'Std energy':<30} {m_base['std_energy']:>15.4f} {m_ksd['std_energy']:>15.4f}")
    print(f"  {'KL divergence':<30} {m_base['kl']:>15.4f} {m_ksd['kl']:>15.4f}")
    print(f"  {'W2 distance':<30} {m_base['w2']:>15.4f} {m_ksd['w2']:>15.4f}")
    print(f"  {'Sinkhorn divergence':<30} {m_base['sinkhorn']:>15.4f} {m_ksd['sinkhorn']:>15.4f}")
    print(f"  {'Mode weight TV':<30} {m_base['mode_weight_tv']:>15.4f} {m_ksd['mode_weight_tv']:>15.4f}")
    print(f"  {'ESS':<30} {m_base['ess']:>15.1f} {m_ksd['ess']:>15.1f}")
    print(f"  {'ESS %':<30} {m_base['ess_pct']:>14.2f}% {m_ksd['ess_pct']:>14.2f}%")
    print("=" * 70)

    # ---- Figures ----
    # Terminal (3-panel)
    print("\n  Generating terminal distribution figure...")
    plot_terminal_neurips(
        energy, ref_for_plot, samples_base, samples_ksd,
        centers, m_base, m_ksd,
        FIG_DIR / 'grid25_terminal_neurips.png',
    )

    # Marginal evolution
    print("  Generating full trajectories...")
    torch.manual_seed(42)
    states_base, ts_base_full = generate_full_states(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(42)
    states_ksd, ts_ksd_full = generate_full_states(sde_ksd, src_ksd, ts_ksd, n_samples, device)

    print("  Generating ASBS marginal evolution...")
    plot_marginal_neurips(
        energy, states_base, ts_base_full, centers, std,
        'ASBS', FIG_DIR / 'grid25_marginal_asbs_neurips.png', C_ASBS,
    )

    print("  Generating SDR-ASBS marginal evolution...")
    plot_marginal_neurips(
        energy, states_ksd, ts_ksd_full, centers, std,
        'SDR-ASBS', FIG_DIR / 'grid25_marginal_sdr_neurips.png', C_SDR,
    )

    # ---- Update 2d_result.md ----
    print("\n  Updating 2d_result.md...")
    md_path = EVAL_DIR / '2d_result.md'
    with open(md_path, 'r') as f:
        content = f.read()

    # Replace existing Grid25 section
    marker_start = '## 25-Mode Grid'
    marker_end = '---\n'  # the trailing ---

    if marker_start in content:
        idx_start = content.index(marker_start)
        # Find the --- after this section
        idx_end = content.index(marker_end, idx_start) + len(marker_end)
        before = content[:idx_start]
        after = content[idx_end:]
    else:
        before = content
        after = ''

    K = m_base['n_modes_total']
    new_section = f"""## 25-Mode Grid (5×5)

| Metric | ASBS (Baseline) | SDR-ASBS (λ={ksd_lambda}) |
|---|---|---|
| Modes covered (of {K}) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} |
| Std energy | {m_base['std_energy']:.4f} | {m_ksd['std_energy']:.4f} |
| KL divergence | {m_base['kl']:.4f} | {m_ksd['kl']:.4f} |
| W₂ distance | {m_base['w2']:.4f} | {m_ksd['w2']:.4f} |
| Sinkhorn divergence | {m_base['sinkhorn']:.4f} | {m_ksd['sinkhorn']:.4f} |
| Mode weight TV | {m_base['mode_weight_tv']:.4f} | {m_ksd['mode_weight_tv']:.4f} |
| ESS | {m_base['ess']:.1f} ({m_base['ess_pct']:.2f}%) | {m_ksd['ess']:.1f} ({m_ksd['ess_pct']:.2f}%) |
| Per-mode counts | {m_base['per_mode_counts']} | {m_ksd['per_mode_counts']} |

### Terminal Distribution

![grid25 terminal](figures_2d/grid25_terminal_neurips.png)

### Marginal Evolution: ASBS

![grid25 marginal asbs](figures_2d/grid25_marginal_asbs_neurips.png)

### Marginal Evolution: SDR-ASBS

![grid25 marginal sdr](figures_2d/grid25_marginal_sdr_neurips.png)

---
"""

    with open(md_path, 'w') as f:
        f.write(before + new_section + after)

    print(f"  Updated {md_path}")
    print("\n=== NeurIPS evaluation complete! ===")


if __name__ == '__main__':
    main()
