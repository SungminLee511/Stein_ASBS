"""
scripts/eval_grid25_baselines.py

Evaluate Grid25 baselines (AS, DGFS, iDEM, pDEM) and update 2d_result.md.
Generates:
  - Unified 6-panel terminal distribution figure (GT, ASBS, SDR-ASBS, AS, DGFS, iDEM/pDEM best)
  - Per-method terminal distribution (individual panels)
  - Metrics table appended to 2d_result.md
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra

# ====================================================================
# Paths
# ====================================================================
PROJ_ROOT = Path('/home/RESEARCH/Stein_ASBS')
RESULTS_DIR = PROJ_ROOT / 'results'
EVAL_DIR = PROJ_ROOT / 'evaluation'
FIG_DIR = EVAL_DIR / 'figures_2d'

XLIM = (-6, 6)
YLIM = (-6, 6)

N_SAMPLES = 2000


# ====================================================================
# Metrics (shared)
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


def compute_w2_distance(samples, ref_samples):
    import ot
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    wa = np.ones(len(a)) / len(a)
    wb = np.ones(len(b)) / len(b)
    M = ot.dist(a, b, metric='sqeuclidean')
    w2_sq = ot.emd2(wa, wb, M)
    return float(np.sqrt(w2_sq))


def compute_sinkhorn_divergence(samples, ref_samples, reg=0.1):
    import ot
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    wa = np.ones(len(a)) / len(a)
    wb = np.ones(len(b)) / len(b)
    M = ot.dist(a, b, metric='sqeuclidean')
    return float(ot.sinkhorn2(wa, wb, M, reg=reg))


def compute_kl_divergence(samples, energy, xlim=XLIM, ylim=YLIM, n_grid=300):
    from scipy.stats import gaussian_kde
    s = samples.cpu().numpy().T
    try:
        kde = gaussian_kde(s, bw_method='silverman')
    except np.linalg.LinAlgError:
        return float('nan')
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
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


def compute_all_metrics(samples, energy, centers, std, ref_samples):
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
    print("    Computing KL...")
    metrics['kl'] = compute_kl_divergence(samples, energy)
    return metrics


# ====================================================================
# 1. Adjoint Sampler (AS) — uses adjoint_samplers framework
# ====================================================================

def load_as_samples(device):
    """Load AS model and generate samples."""
    from adjoint_samplers.components.sde import ControlledSDE, sdeint
    import adjoint_samplers.utils.train_utils as train_utils

    exp_dir = RESULTS_DIR / 'grid25_as' / 'seed_0'
    cfg_path = exp_dir / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)

    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Find best checkpoint (highest epoch number)
    ckpt_dir = exp_dir / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('checkpoint_*.pt'))
    # Prefer checkpoint_latest.pt, else last numbered
    ckpt_path = ckpt_dir / 'checkpoint_latest.pt'
    if not ckpt_path.exists():
        ckpt_path = ckpts[-1]
    print(f"  AS checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    ts_cfg = {
        't0': float(cfg.timesteps.t0),
        't1': float(cfg.timesteps.t1),
        'steps': int(cfg.timesteps.steps),
        'rescale_t': cfg.timesteps.rescale_t if cfg.timesteps.rescale_t is not None else None,
    }

    torch.manual_seed(0)
    x0 = source.sample([N_SAMPLES]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    with torch.no_grad():
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)

    return x1, energy


# ====================================================================
# 2. DGFS (GFlowNet) — uses gflownet framework
# ====================================================================

def load_dgfs_samples(device):
    """Load DGFS model and generate samples."""
    sys.path.insert(0, str(PROJ_ROOT / 'baseline_models' / 'dgfs'))

    from gflownet.gflownet import get_alg, sample_traj
    from target.distribution.grid25 import Grid25

    exp_dir = RESULTS_DIR / 'grid25_dgfs'
    cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')

    # Resolve interpolations and add missing keys
    cfg.data_ndim = 2
    cfg.t_end = cfg.dt * cfg.N
    if not OmegaConf.is_missing(cfg, 'wd'):
        cfg.weight_decay = cfg.wd
    elif 'weight_decay' not in cfg:
        cfg.weight_decay = 1e-7
    if 'subtb_lambda' not in cfg:
        cfg.subtb_lambda = cfg.get('stlam', 2.0)
    if 'batch_size' not in cfg:
        cfg.batch_size = cfg.get('bs', 256)

    # Instantiate target (energy function)
    data = Grid25()

    # Create GFlowNet model
    model = get_alg(cfg, task=data).to(device)

    # Load checkpoint
    ckpt_path = exp_dir / 'checkpoint_best.pt'
    print(f"  DGFS checkpoint: {ckpt_path.name}")
    model.load(str(ckpt_path))
    model.eval()

    # Sample
    logreward_fn = lambda x: -data.energy(x)
    torch.manual_seed(0)
    with torch.no_grad():
        traj, info = sample_traj(model, cfg, logreward_fn, batch_size=N_SAMPLES)

    # Terminal samples are the last element of trajectory
    samples = traj[-1][1].to(device)  # (N, 2)
    return samples


# ====================================================================
# 3. DEM (iDEM / pDEM) — uses DEM Lightning framework
# ====================================================================

def load_dem_samples(exp_name, device):
    """Load DEM (iDEM or pDEM) model and generate samples.

    Args:
        exp_name: 'grid25_idem' or 'grid25_pdem'
    """
    sys.path.insert(0, str(PROJ_ROOT / 'baseline_models' / 'dem'))

    from dem.models.dem_module import DEMLitModule
    from dem.models.components.sdes import VEReverseSDE
    from dem.models.components.sde_integration import integrate_sde
    from dem.energies.base_prior import Prior

    exp_dir = RESULTS_DIR / exp_name
    cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')

    # Instantiate energy function
    energy_function = hydra.utils.instantiate(cfg.energy)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, energy_function=energy_function)

    # Load checkpoint state_dict
    ckpt_path = exp_dir / 'checkpoints' / 'last.ckpt'
    print(f"  {exp_name} checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    # Manually set up prior (normally done in setup())
    noise_schedule = model.noise_schedule
    scale = noise_schedule.h(torch.tensor(1.0)).item() ** 0.5
    model.prior = Prior(dim=energy_function.dimensionality, device=device, scale=scale)

    # Generate samples using the model's reverse_sde (initialized in __init__)
    torch.manual_seed(0)
    with torch.no_grad():
        samples = model.generate_samples(
            num_samples=N_SAMPLES,
            diffusion_scale=model.diffusion_scale,
        )

    return samples


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
    ax.contourf(xx.numpy(), yy.numpy(), p.numpy(), levels=25, cmap='Blues', alpha=0.5)
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(), levels=8,
               colors='#2c3e50', alpha=0.25, linewidths=0.3)


def plot_baselines_terminal(
    energy, centers, all_samples, all_names, all_metrics, output_path,
):
    """Multi-panel terminal distribution figure for all baselines."""
    n_methods = len(all_names)
    # Include GT panel
    n_panels = 1 + n_methods
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    colors = {
        'ASBS': '#c0392b',
        'SDR-ASBS': '#e67e22',
        'AS': '#2ecc71',
        'DGFS': '#3498db',
        'iDEM': '#9b59b6',
        'pDEM': '#1abc9c',
    }

    # Flatten axes
    flat_axes = axes.flatten()

    # Panel 0: GT
    ax = flat_axes[0]
    ax.set_facecolor('#fafafa')
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
    plot_density_contours(ax, energy, XLIM, YLIM)
    c = centers.cpu().numpy()
    ax.scatter(c[:, 0], c[:, 1], marker='+', s=20, c='black', linewidths=0.5, zorder=10)
    ref_samples = energy.get_ref_samples(N_SAMPLES)
    s = ref_samples.numpy()
    ax.scatter(s[:, 0], s[:, 1], s=2, c='#555555', alpha=0.35, zorder=5, edgecolors='none')
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')

    # Method panels
    for i, (name, samples, metrics) in enumerate(zip(all_names, all_samples, all_metrics)):
        ax = flat_axes[i + 1]
        ax.set_facecolor('#fafafa')
        ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')
        plot_density_contours(ax, energy, XLIM, YLIM)
        ax.scatter(c[:, 0], c[:, 1], marker='+', s=20, c='black', linewidths=0.5, zorder=10)
        s = samples.cpu().numpy()
        color = colors.get(name, '#888888')
        ax.scatter(s[:, 0], s[:, 1], s=2, c=color, alpha=0.35, zorder=5, edgecolors='none')
        K = metrics['n_modes_total']
        n_cov = metrics['n_modes_covered']
        ax.set_title(f"{name}\n({n_cov}/{K} modes, W₂={metrics['w2']:.2f})",
                     fontsize=11, fontweight='bold')

    # Hide unused axes
    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle('25-Mode Grid (5×5): Baseline Comparison', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Grid25 Baseline Evaluation")
    print("  Methods: AS, DGFS, iDEM, pDEM")
    print("=" * 60)

    # We need the energy function for metrics. Load from ASBS config.
    from adjoint_samplers.components.sde import ControlledSDE, sdeint
    asbs_cfg = OmegaConf.load(RESULTS_DIR / 'grid25_asbs' / 'seed_0' / '.hydra' / 'config.yaml')
    energy = hydra.utils.instantiate(asbs_cfg.energy, device='cpu')
    centers = energy.get_centers()
    std = energy.get_std()
    ref_samples = energy.get_ref_samples(5000).to(device)

    # Also load existing ASBS and SDR-ASBS for the combined figure
    print("\n--- Loading ASBS (existing) ---")
    import adjoint_samplers.utils.train_utils as train_utils

    def _load_asbs(exp_dir_name):
        exp_dir = RESULTS_DIR / exp_dir_name / 'seed_0'
        cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')
        e = hydra.utils.instantiate(cfg.energy, device=device)
        src = hydra.utils.instantiate(cfg.source, device=device)
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        ctrl = hydra.utils.instantiate(cfg.controller).to(device)
        sde = ControlledSDE(ref_sde, ctrl).to(device)
        ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ctrl.load_state_dict(ckpt['controller'])
        ts_cfg = {
            't0': float(cfg.timesteps.t0), 't1': float(cfg.timesteps.t1),
            'steps': int(cfg.timesteps.steps),
            'rescale_t': cfg.timesteps.rescale_t if cfg.timesteps.rescale_t is not None else None,
        }
        torch.manual_seed(0)
        x0 = src.sample([N_SAMPLES]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        with torch.no_grad():
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        return x1

    samples_asbs = _load_asbs('grid25_asbs')
    print("  ASBS loaded.")
    samples_sdr = _load_asbs('grid25_ksd_asbs_lam01')
    print("  SDR-ASBS loaded.")

    # ---- Load baselines ----
    all_samples = [samples_asbs, samples_sdr]
    all_names = ['ASBS', 'SDR-ASBS']

    # AS
    print("\n--- Loading AS (Adjoint Sampler) ---")
    samples_as, _ = load_as_samples(device)
    all_samples.append(samples_as)
    all_names.append('AS')
    print(f"  AS: {samples_as.shape}")

    # DGFS
    print("\n--- Loading DGFS ---")
    samples_dgfs = load_dgfs_samples(device)
    all_samples.append(samples_dgfs)
    all_names.append('DGFS')
    print(f"  DGFS: {samples_dgfs.shape}")

    # iDEM
    print("\n--- Loading iDEM ---")
    samples_idem = load_dem_samples('grid25_idem', device)
    all_samples.append(samples_idem)
    all_names.append('iDEM')
    print(f"  iDEM: {samples_idem.shape}")

    # pDEM
    print("\n--- Loading pDEM ---")
    samples_pdem = load_dem_samples('grid25_pdem', device)
    all_samples.append(samples_pdem)
    all_names.append('pDEM')
    print(f"  pDEM: {samples_pdem.shape}")

    # ---- Compute metrics for all ----
    all_metrics = []
    for name, samples in zip(all_names, all_samples):
        print(f"\n  === {name} Metrics ===")
        m = compute_all_metrics(samples, energy, centers, std, ref_samples)
        all_metrics.append(m)
        K = m['n_modes_total']
        print(f"    Modes: {m['n_modes_covered']}/{K}")
        print(f"    Mean E: {m['mean_energy']:.4f}, Std E: {m['std_energy']:.4f}")
        print(f"    W2: {m['w2']:.4f}, Sinkhorn: {m['sinkhorn']:.4f}")
        print(f"    KL: {m['kl']:.4f}, Mode TV: {m['mode_weight_tv']:.4f}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    header = f"  {'Metric':<25}"
    for name in all_names:
        header += f" {name:>12}"
    print(header)
    print("-" * 100)

    for metric_name, key in [
        ('Modes covered', 'n_modes_covered'),
        ('Mean energy', 'mean_energy'),
        ('Std energy', 'std_energy'),
        ('KL divergence', 'kl'),
        ('W₂ distance', 'w2'),
        ('Sinkhorn div.', 'sinkhorn'),
        ('Mode weight TV', 'mode_weight_tv'),
    ]:
        row = f"  {metric_name:<25}"
        for m in all_metrics:
            v = m[key]
            if isinstance(v, int):
                row += f" {v:>12}"
            else:
                row += f" {v:>12.4f}"
        print(row)
    print("=" * 100)

    # ---- Generate figure ----
    print("\n  Generating baseline comparison figure...")
    plot_baselines_terminal(
        energy, centers, all_samples, all_names, all_metrics,
        FIG_DIR / 'grid25_baselines_terminal.png',
    )

    # ---- Update 2d_result.md ----
    print("\n  Updating 2d_result.md...")
    md_path = EVAL_DIR / '2d_result.md'
    with open(md_path, 'r') as f:
        content = f.read()

    kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

    # Build baseline section
    # Table header
    header_cols = " | ".join(all_names)
    table = f"| Metric | {header_cols} |\n"
    table += "|---" * (len(all_names) + 1) + "|\n"

    for metric_name, key, fmt in [
        ('Modes covered (of 25)', 'n_modes_covered', 'd'),
        ('Mean energy', 'mean_energy', '.4f'),
        ('Std energy', 'std_energy', '.4f'),
        ('KL divergence', 'kl', '.4f'),
        ('W₂ distance', 'w2', '.4f'),
        ('Sinkhorn divergence', 'sinkhorn', '.4f'),
        ('Mode weight TV', 'mode_weight_tv', '.4f'),
    ]:
        row = f"| {metric_name}"
        for m in all_metrics:
            v = m[key]
            row += f" | {v:{fmt}}"
        row += " |\n"
        table += row

    new_section = f"""## 25-Mode Grid — Baseline Comparison

Evaluated: {kst}

Methods: ASBS, SDR-ASBS (λ=0.1), AS (Adjoint Sampler), DGFS (GFlowNet), iDEM, pDEM

{table}
### Terminal Distribution Comparison

![grid25 baselines](figures_2d/grid25_baselines_terminal.png)

---
"""

    # Insert before the trailing empty line or at end
    # Check if section already exists
    marker = '## 25-Mode Grid — Baseline Comparison'
    if marker in content:
        idx_start = content.index(marker)
        # Find the --- after this section
        try:
            idx_end = content.index('---\n', idx_start) + len('---\n')
            content = content[:idx_start] + new_section + content[idx_end:]
        except ValueError:
            content = content[:idx_start] + new_section
    else:
        content += '\n' + new_section

    with open(md_path, 'w') as f:
        f.write(content)

    print(f"  Updated {md_path}")
    print("\n=== Baseline evaluation complete! ===")


if __name__ == '__main__':
    main()
