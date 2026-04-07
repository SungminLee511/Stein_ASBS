"""
scripts/eval_grid25_seeds.py

Evaluate all 3 ASBS seeds on Grid25, compute metrics, generate NeurIPS figures,
and update evaluation/2d_result.md.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
import ot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils

# ====================================================================
# NeurIPS style (copied from eval_grid25_neurips.py)
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
        'lines.markersize': 3,
        'axes.linewidth': 0.6,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
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

C_REF = '#555555'
C_ASBS = '#c0392b'
C_SDR = '#e67e22'
C_CONTOUR = '#2c3e50'
C_BG = '#fafafa'

# Distinct colors for 3 seeds
C_SEEDS = ['#c0392b', '#2980b9', '#27ae60']  # red, blue, green

TEXT_WIDTH_IN = 5.5

SEEDS = [0, 1, 2]
N_SAMPLES = 2000


# ====================================================================
# Loading & Sampling
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
# Metrics (copied from eval_grid25_neurips.py)
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
    K = len(counts)
    total = sum(counts)
    if total == 0:
        return 1.0
    empirical = np.array(counts, dtype=np.float64) / total
    uniform = np.ones(K) / K
    return 0.5 * np.abs(empirical - uniform).sum()


def compute_kl_divergence(samples, energy, n_grid=300):
    from scipy.stats import gaussian_kde
    s = samples.cpu().numpy().T
    try:
        kde = gaussian_kde(s, bw_method='silverman')
    except np.linalg.LinAlgError:
        return float('nan')
    x = np.linspace(XLIM[0], XLIM[1], n_grid)
    y = np.linspace(YLIM[0], YLIM[1], n_grid)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=0)
    log_q = kde.logpdf(grid_pts)
    grid_tensor = torch.tensor(grid_pts.T, dtype=torch.float32)
    with torch.no_grad():
        E = energy.eval(grid_tensor).cpu().numpy()
    log_p_unnorm = -E
    log_Z = np.log(np.exp(log_p_unnorm - log_p_unnorm.max()).sum()) + log_p_unnorm.max()
    dx = (x[1] - x[0]) * (y[1] - y[0])
    log_Z += np.log(dx)
    log_p = log_p_unnorm - log_Z
    q_density = np.exp(log_q)
    valid = q_density > 1e-10
    kl = np.sum(q_density[valid] * (log_q[valid] - log_p[valid])) * dx
    return float(kl)


def compute_w2_distance(samples, ref_samples):
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    w2_sq = ot.emd2(wa, wb, M)
    return float(np.sqrt(w2_sq))


def compute_sinkhorn_divergence(samples, ref_samples, reg=0.1):
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    sinkhorn = ot.sinkhorn2(wa, wb, M, reg=reg)
    return float(sinkhorn)


def compute_ess(log_weights):
    log_w = log_weights - torch.logsumexp(log_weights, dim=0)
    ess = torch.exp(-torch.logsumexp(2 * log_w, dim=0))
    return ess.item()


def compute_ess_from_samples(samples, energy, source):
    with torch.no_grad():
        log_p_target = -energy.eval(samples)
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
    }

    print("    Computing W2...")
    metrics['w2'] = compute_w2_distance(samples, ref_samples)
    print("    Computing Sinkhorn...")
    metrics['sinkhorn'] = compute_sinkhorn_divergence(samples, ref_samples)
    print("    Computing KL divergence...")
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
# Plotting
# ====================================================================

def plot_density_contours(ax, energy, xlim, ylim, n_grid=250):
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


def plot_terminal_single_seed(
    energy, samples_ref, samples_asbs, centers,
    m_asbs, seed_idx, output_path, color,
):
    """2-panel terminal: Ground Truth + ASBS Seed N."""
    fig_h = TEXT_WIDTH_IN / 2 * 0.95
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN * 0.67, fig_h))

    titles = [
        r'Ground Truth',
        f'ASBS Seed {seed_idx} ({m_asbs["n_modes_covered"]}/{m_asbs["n_modes_total"]})',
    ]
    sample_sets = [samples_ref, samples_asbs]
    colors = [C_REF, color]

    for ax, title, samples, c in zip(axes, titles, sample_sets, colors):
        ax.set_facecolor(C_BG)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        plot_density_contours(ax, energy, XLIM, YLIM)
        ctr = centers.cpu().numpy()
        ax.scatter(ctr[:, 0], ctr[:, 1], marker='+', s=15, c='black',
                   linewidths=0.5, zorder=10)
        s = samples.cpu().numpy()
        ax.scatter(s[:, 0], s[:, 1], s=1.5, c=c, alpha=0.35, zorder=5,
                   edgecolors='none', rasterized=True)
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
    print(f"  Saved: {output_path}")


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
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='+', s=12, c='black',
                   linewidths=0.4, zorder=10)
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
    fig.text(0.01, 0.5, method_name, fontsize=10, fontweight='bold',
             va='center', ha='left', rotation=90)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    set_neurips_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Grid25 ASBS — 3-Seed Evaluation + NeurIPS Figures")
    print("=" * 70)

    # Collect all seed metrics
    all_metrics = {}
    all_samples = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        seed_dir = RESULTS_DIR / 'grid25_asbs' / f'seed_{seed}'
        print(f"  Loading model from {seed_dir}...")
        sde, source, energy, ts_cfg = load_model(seed_dir, device)

        centers = energy.get_centers().to(device)
        std = energy.get_std()

        # Generate samples (use seed for reproducibility)
        torch.manual_seed(seed * 1000)
        print(f"  Generating {N_SAMPLES} terminal samples...")
        samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, device)
        all_samples[seed] = samples

        # Reference samples
        ref_samples = energy.get_ref_samples(5000).to(device)
        ref_for_plot = ref_samples[:N_SAMPLES]

        # Metrics
        print(f"  Computing metrics...")
        m = compute_all_metrics(samples, energy, centers, std, ref_samples, source)
        all_metrics[seed] = m

        # Print
        print(f"\n  --- Seed {seed} Results ---")
        print(f"  Modes covered: {m['n_modes_covered']}/{m['n_modes_total']}")
        print(f"  Mean energy:   {m['mean_energy']:.4f}")
        print(f"  Std energy:    {m['std_energy']:.4f}")
        print(f"  KL:            {m['kl']:.4f}")
        print(f"  W2:            {m['w2']:.4f}")
        print(f"  Sinkhorn:      {m['sinkhorn']:.4f}")
        print(f"  Mode TV:       {m['mode_weight_tv']:.4f}")
        print(f"  ESS:           {m['ess']:.1f} ({m['ess_pct']:.2f}%)")

        # ---- Figure 1: Terminal distribution ----
        print(f"\n  Generating terminal figure for seed {seed}...")
        plot_terminal_single_seed(
            energy, ref_for_plot, samples, centers,
            m, seed,
            FIG_DIR / f'grid25_asbs_seed{seed}_terminal_neurips.png',
            C_SEEDS[seed],
        )

        # ---- Figure 2: Marginal evolution ----
        print(f"  Generating marginal evolution for seed {seed}...")
        torch.manual_seed(seed * 1000 + 42)
        states, ts_full = generate_full_states(sde, source, ts_cfg, N_SAMPLES, device)
        plot_marginal_neurips(
            energy, states, ts_full, centers, std,
            f'ASBS (seed {seed})',
            FIG_DIR / f'grid25_asbs_seed{seed}_marginal_neurips.png',
            C_SEEDS[seed],
        )

        # ---- Figure 3: Terminal with density heatmap overlay ----
        print(f"  Generating density heatmap for seed {seed}...")
        fig_h = TEXT_WIDTH_IN / 3 * 0.95
        fig, ax = plt.subplots(1, 1, figsize=(fig_h * 1.1, fig_h))
        ax.set_facecolor(C_BG)

        # KDE heatmap of samples
        from scipy.stats import gaussian_kde
        s_np = samples.cpu().numpy().T
        try:
            kde = gaussian_kde(s_np, bw_method='silverman')
            xg = np.linspace(XLIM[0], XLIM[1], 200)
            yg = np.linspace(YLIM[0], YLIM[1], 200)
            xxg, yyg = np.meshgrid(xg, yg)
            grid_pts = np.stack([xxg.ravel(), yyg.ravel()], axis=0)
            z = kde(grid_pts).reshape(200, 200)
            ax.contourf(xxg, yyg, z, levels=30, cmap='Reds', alpha=0.7)
            ax.contour(xxg, yyg, z, levels=10, colors='darkred', alpha=0.3, linewidths=0.3)
        except Exception:
            pass

        # Plot density contours of target
        plot_density_contours(ax, energy, XLIM, YLIM)

        ctr = centers.cpu().numpy()
        ax.scatter(ctr[:, 0], ctr[:, 1], marker='+', s=20, c='black',
                   linewidths=0.6, zorder=10)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect('equal')
        ax.set_title(f'ASBS Seed {seed} — KDE', fontsize=9, pad=4)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(length=2, width=0.4)
        ax.set_xlabel(r'$x_1$', fontsize=9)
        ax.set_ylabel(r'$x_2$', fontsize=9)

        kde_path = FIG_DIR / f'grid25_asbs_seed{seed}_kde_neurips.png'
        fig.savefig(kde_path, dpi=200, bbox_inches='tight')
        fig.savefig(kde_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {kde_path}")

        # Free GPU memory
        del sde, source, states
        torch.cuda.empty_cache()

    # ====================================================================
    # Summary table
    # ====================================================================
    print("\n" + "=" * 80)
    print(f"  {'Metric':<25} {'Seed 0':>15} {'Seed 1':>15} {'Seed 2':>15}")
    print("-" * 80)
    for key, label in [
        ('n_modes_covered', 'Modes covered'),
        ('mean_energy', 'Mean energy'),
        ('std_energy', 'Std energy'),
        ('kl', 'KL divergence'),
        ('w2', 'W2 distance'),
        ('sinkhorn', 'Sinkhorn div'),
        ('mode_weight_tv', 'Mode weight TV'),
        ('ess', 'ESS'),
        ('ess_pct', 'ESS %'),
    ]:
        vals = []
        for seed in SEEDS:
            v = all_metrics[seed][key]
            if key == 'n_modes_covered':
                vals.append(f"{v}/25")
            elif key == 'ess_pct':
                vals.append(f"{v:.2f}%")
            elif key == 'ess':
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.4f}")
        print(f"  {label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")
    print("=" * 80)

    # Compute mean ± std across seeds
    print("\n  Mean ± Std across seeds:")
    for key, label in [
        ('mean_energy', 'Mean energy'),
        ('kl', 'KL divergence'),
        ('w2', 'W2 distance'),
        ('sinkhorn', 'Sinkhorn div'),
        ('mode_weight_tv', 'Mode weight TV'),
        ('ess', 'ESS'),
    ]:
        vals = [all_metrics[s][key] for s in SEEDS]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        print(f"    {label:<25} {mean_v:.4f} ± {std_v:.4f}")

    # ====================================================================
    # Update 2d_result.md
    # ====================================================================
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')

    md_path = EVAL_DIR / '2d_result.md'
    with open(md_path, 'r') as f:
        content = f.read()

    # Build new section for 3-seed results
    # Insert BEFORE the baseline comparison section
    new_section = f"""## 25-Mode Grid — ASBS 3-Seed Evaluation

Evaluated: {now_kst}

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---|---|---|---|
"""

    for key, label in [
        ('n_modes_covered', 'Modes covered (of 25)'),
        ('mean_energy', 'Mean energy'),
        ('std_energy', 'Std energy'),
        ('kl', 'KL divergence'),
        ('w2', 'W₂ distance'),
        ('sinkhorn', 'Sinkhorn divergence'),
        ('mode_weight_tv', 'Mode weight TV'),
        ('ess', 'ESS'),
    ]:
        cells = []
        vals_for_stats = []
        for s in SEEDS:
            v = all_metrics[s][key]
            vals_for_stats.append(v)
            if key == 'n_modes_covered':
                cells.append(f"{v}")
            elif key == 'ess':
                pct = all_metrics[s]['ess_pct']
                cells.append(f"{v:.1f} ({pct:.2f}%)")
            else:
                cells.append(f"{v:.4f}")

        mean_v = np.mean(vals_for_stats)
        std_v = np.std(vals_for_stats)
        if key == 'n_modes_covered':
            mean_std = f"{mean_v:.1f} ± {std_v:.1f}"
        elif key == 'ess':
            mean_pct = np.mean([all_metrics[s]['ess_pct'] for s in SEEDS])
            std_pct = np.std([all_metrics[s]['ess_pct'] for s in SEEDS])
            mean_std = f"{mean_v:.1f} ({mean_pct:.2f}% ± {std_pct:.2f}%)"
        else:
            mean_std = f"{mean_v:.4f} ± {std_v:.4f}"

        new_section += f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_std} |\n"

    # Add per-mode counts
    new_section += f"| Per-mode counts (seed 0) | {all_metrics[0]['per_mode_counts']} | | | |\n"
    new_section += f"| Per-mode counts (seed 1) | | {all_metrics[1]['per_mode_counts']} | | |\n"
    new_section += f"| Per-mode counts (seed 2) | | | {all_metrics[2]['per_mode_counts']} | |\n"

    # Add figures
    new_section += """
### Seed 0

#### Terminal Distribution
![grid25 asbs seed0 terminal](figures_2d/grid25_asbs_seed0_terminal_neurips.png)

#### Marginal Evolution
![grid25 asbs seed0 marginal](figures_2d/grid25_asbs_seed0_marginal_neurips.png)

#### KDE Density
![grid25 asbs seed0 kde](figures_2d/grid25_asbs_seed0_kde_neurips.png)

### Seed 1

#### Terminal Distribution
![grid25 asbs seed1 terminal](figures_2d/grid25_asbs_seed1_terminal_neurips.png)

#### Marginal Evolution
![grid25 asbs seed1 marginal](figures_2d/grid25_asbs_seed1_marginal_neurips.png)

#### KDE Density
![grid25 asbs seed1 kde](figures_2d/grid25_asbs_seed1_kde_neurips.png)

### Seed 2

#### Terminal Distribution
![grid25 asbs seed2 terminal](figures_2d/grid25_asbs_seed2_terminal_neurips.png)

#### Marginal Evolution
![grid25 asbs seed2 marginal](figures_2d/grid25_asbs_seed2_marginal_neurips.png)

#### KDE Density
![grid25 asbs seed2 kde](figures_2d/grid25_asbs_seed2_kde_neurips.png)

---
"""

    # Insert before "## 25-Mode Grid — Baseline Comparison"
    marker = '## 25-Mode Grid — Baseline Comparison'
    if marker in content:
        idx = content.index(marker)
        content = content[:idx] + new_section + '\n' + content[idx:]
    else:
        # Append before final ---
        content = content.rstrip()
        if content.endswith('---'):
            content = content[:-3]
        content += '\n' + new_section

    with open(md_path, 'w') as f:
        f.write(content)

    print(f"\n  Updated {md_path}")
    print("\n=== 3-Seed evaluation complete! ===")


if __name__ == '__main__':
    main()
