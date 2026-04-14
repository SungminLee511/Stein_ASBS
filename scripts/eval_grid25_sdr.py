"""
scripts/eval_grid25_sdr.py

Evaluate Grid25: Vanilla ASBS vs SDR β={0.5, 0.7, 1.0}
Metrics: Mode Weight TV, Energy W2, Sinkhorn, W2, KL, Std Energy
Figure: Single stacked marginal evolution (NeurIPS style)
"""

import torch
import numpy as np
import json
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
# NeurIPS style
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

RESULTS_DIR = Path('/home/sky/SML/Stein_ASBS/results')
EVAL_DIR = Path('/home/sky/SML/Stein_ASBS/evaluation')
FIG_DIR = EVAL_DIR / 'figures_2d'

XLIM = (-6, 6)
YLIM = (-6, 6)

C_REF = '#555555'
C_ASBS = '#c0392b'
C_SDR05 = '#2980b9'   # blue
C_SDR07 = '#27ae60'   # green
C_SDR10 = '#8e44ad'   # purple
C_CONTOUR = '#2c3e50'
C_BG = '#fafafa'

TEXT_WIDTH_IN = 5.5


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
# Metrics (no mode coverage)
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
    return assignments, counts


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
    return float(np.sqrt(max(w2_sq, 0)))


def compute_sinkhorn_divergence(samples, ref_samples, reg=1.0):
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M_pq = ot.dist(a, b, metric='sqeuclidean')
    ot_pq = ot.sinkhorn2(wa, wb, M_pq, reg=reg)
    M_pp = ot.dist(a, a, metric='sqeuclidean')
    ot_pp = ot.sinkhorn2(wa, wa, M_pp, reg=reg)
    M_qq = ot.dist(b, b, metric='sqeuclidean')
    ot_qq = ot.sinkhorn2(wb, wb, M_qq, reg=reg)
    return float(max(ot_pq - 0.5 * ot_pp - 0.5 * ot_qq, 0.0))


def compute_energy_w2(samples, ref_samples, energy):
    with torch.no_grad():
        E_gen = energy.eval(samples).cpu().numpy()
        E_ref = energy.eval(ref_samples[:len(samples)]).cpu().numpy()
    E_gen_sorted = np.sort(E_gen)
    E_ref_sorted = np.sort(E_ref)
    n = min(len(E_gen_sorted), len(E_ref_sorted))
    w2 = np.sqrt(np.mean((E_gen_sorted[:n] - E_ref_sorted[:n])**2))
    return float(w2)


def compute_all_metrics(samples, energy, centers, std, ref_samples):
    assignments, counts = assign_modes(samples, centers.to(samples.device), std=std)
    E_gen = energy.eval(samples)
    counts_list = counts.cpu().tolist()

    metrics = {
        'std_energy': E_gen.std().item(),
        'mode_weight_tv': compute_mode_weight_tv(counts_list),
        'energy_w2': compute_energy_w2(samples, ref_samples, energy),
        'w2': compute_w2_distance(samples, ref_samples),
        'sinkhorn': compute_sinkhorn_divergence(samples, ref_samples),
    }

    print("    Computing KL divergence...")
    metrics['kl'] = compute_kl_divergence(samples, energy)

    return metrics


# ====================================================================
# Marginal Evolution Figure (single massive stacked plot)
# ====================================================================

def plot_marginal_stacked(
    energy, all_states, all_ts, centers,
    method_names, method_colors, output_path,
    n_snapshots=6,
):
    """
    One massive figure: rows = methods, cols = time snapshots.
    Row 0 = Reference (GT), then one row per method.
    """
    n_methods = len(method_names)
    n_rows = n_methods  # no GT row — we show GT density as contour bg

    panel_size = 1.3
    fig_w = n_snapshots * panel_size + 1.0  # extra for row labels
    fig_h = n_rows * panel_size
    fig, axes = plt.subplots(n_rows, n_snapshots, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx in range(n_rows):
        states = all_states[row_idx]
        ts = all_ts[row_idx]
        color = method_colors[row_idx]
        name = method_names[row_idx]
        T = len(states)
        indices = np.linspace(0, T - 1, n_snapshots, dtype=int)

        for col_idx, state_idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            t_val = ts[state_idx].item()
            samples = states[state_idx].cpu().numpy()

            ax.set_facecolor(C_BG)

            # Mode center markers
            c = centers.cpu().numpy()
            ax.scatter(c[:, 0], c[:, 1], marker='+', s=10, c='black',
                       linewidths=0.3, zorder=10)

            # Samples
            ax.scatter(samples[:, 0], samples[:, 1], s=1.0, c=color,
                       alpha=0.35, zorder=5, edgecolors='none', rasterized=True)

            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)
            ax.set_aspect('equal')

            # Titles on top row only
            if row_idx == 0:
                ax.set_title(f'$t = {t_val:.2f}$', fontsize=8, pad=3)

            # Ticks
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.tick_params(length=2, width=0.4, labelsize=6)

            # Only left column gets y-label
            if col_idx == 0:
                ax.set_ylabel(name, fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])

            # Only bottom row gets x-labels
            if row_idx < n_rows - 1:
                ax.set_xticklabels([])

    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path} + .pdf")


def _plot_density_bg(ax, energy, n_grid=200):
    x = torch.linspace(XLIM[0], XLIM[1], n_grid)
    y = torch.linspace(YLIM[0], YLIM[1], n_grid)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        E = energy.eval(grid).reshape(n_grid, n_grid)
    log_p = -E
    log_p = log_p - log_p.max()
    p = torch.exp(log_p)
    ax.contourf(xx.numpy(), yy.numpy(), p.numpy(), levels=20,
                cmap='Blues', alpha=0.4, zorder=1)
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(), levels=6,
               colors=C_CONTOUR, alpha=0.2, linewidths=0.25, zorder=2)


# ====================================================================
# Main
# ====================================================================

def main():
    set_neurips_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 2000
    traj_seed = 42

    # Define methods and their experiment directories
    methods = {
        'ASBS': {
            'seeds': [
                RESULTS_DIR / 'grid25_asbs' / 'seed_0',
                RESULTS_DIR / 'grid25_asbs' / 'seed_1',
                RESULTS_DIR / 'grid25_asbs' / 'seed_2',
            ],
            'color': C_ASBS,
        },
        r'SDR $\beta$=0.5': {
            'seeds': [
                RESULTS_DIR / 'grid25_darw_b0.5_s0' / 'seed_0',
                RESULTS_DIR / 'grid25_darw_b0.5_s1' / 'seed_1',
                RESULTS_DIR / 'grid25_darw_b0.5_s2' / 'seed_2',
            ],
            'color': C_SDR05,
        },
        r'SDR $\beta$=0.7': {
            'seeds': [
                RESULTS_DIR / 'grid25_darw_b0.7_s0' / 'seed_0',
                RESULTS_DIR / 'grid25_darw_b0.7_s1' / 'seed_1',
                RESULTS_DIR / 'grid25_darw_b0.7_s2' / 'seed_2',
            ],
            'color': C_SDR07,
        },
        r'SDR $\beta$=1.0': {
            'seeds': [
                RESULTS_DIR / 'grid25_sdr_b1.0_s0' / 'seed_0',
                RESULTS_DIR / 'grid25_sdr_b1.0_s1' / 'seed_1',
                RESULTS_DIR / 'grid25_sdr_b1.0_s2' / 'seed_2',
            ],
            'color': C_SDR10,
        },
    }

    print("=" * 70)
    print("  Grid25 — SDR Evaluation (NeurIPS style)")
    print("=" * 70)

    # Load energy from first available model
    first_dir = methods['ASBS']['seeds'][0]
    _, _, energy, _ = load_model(first_dir, device)
    centers = energy.get_centers().to(device)
    std = energy.get_std()
    ref_samples = energy.get_ref_samples(5000).to(device)

    # ---- Compute metrics per method (mean ± std over seeds) ----
    all_results = {}

    for method_name, method_info in methods.items():
        print(f"\n  === {method_name} ===")
        seed_metrics = []

        for seed_dir in method_info['seeds']:
            print(f"    Seed: {seed_dir.name} ({seed_dir.parent.name})")
            sde, source, _, ts_cfg = load_model(seed_dir, device)

            torch.manual_seed(0)
            samples = generate_samples(sde, source, ts_cfg, n_samples, device)
            m = compute_all_metrics(samples, energy, centers, std, ref_samples)
            seed_metrics.append(m)

            # Free GPU memory
            del sde, source, samples
            torch.cuda.empty_cache()

        # Aggregate: mean ± std
        keys = seed_metrics[0].keys()
        agg = {}
        for k in keys:
            vals = [m[k] for m in seed_metrics if not (isinstance(m[k], float) and np.isnan(m[k]))]
            if len(vals) > 0:
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
            else:
                agg[f'{k}_mean'] = float('nan')
                agg[f'{k}_std'] = float('nan')
        all_results[method_name] = agg

    # ---- Print results table ----
    print("\n" + "=" * 100)
    metric_keys = ['mode_weight_tv', 'energy_w2', 'w2', 'sinkhorn', 'kl']
    metric_labels = {
        'mode_weight_tv': 'Mode Weight TV ↓',
        'energy_w2': 'Energy W2 ↓',
        'w2': 'W2 Distance ↓',
        'sinkhorn': 'Sinkhorn Div ↓',
        'kl': 'KL Divergence ↓',
    }

    method_names = list(methods.keys())
    header = f"  {'Metric':<35}" + "".join(f"{m:>22}" for m in method_names)
    print(header)
    print("-" * 100)

    for k in metric_keys:
        row = f"  {metric_labels[k]:<35}"
        for m_name in method_names:
            mean_val = all_results[m_name][f'{k}_mean']
            std_val = all_results[m_name][f'{k}_std']
            if np.isnan(mean_val):
                row += f"{'N/A':>22}"
            else:
                row += f"{mean_val:>12.4f}±{std_val:<8.4f}"
        print(row)
    print("=" * 100)

    # ---- Save metrics JSON ----
    json_path = EVAL_DIR / 'grid25_sdr_results.json'
    # Convert for JSON serialization
    json_results = {}
    for m_name, agg in all_results.items():
        json_results[m_name] = {k: v for k, v in agg.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved metrics: {json_path}")

    # ---- Generate marginal evolution figure ----
    print("\n  Generating marginal evolution trajectories...")

    # Use seed 0 for each method for the figure
    all_states_list = []
    all_ts_list = []
    fig_method_names = []
    fig_method_colors = []

    for method_name, method_info in methods.items():
        seed_dir = method_info['seeds'][0]  # seed 0
        print(f"    Generating trajectories for {method_name} (seed 0)...")
        sde, source, _, ts_cfg = load_model(seed_dir, device)

        torch.manual_seed(traj_seed)
        states, ts = generate_full_states(sde, source, ts_cfg, n_samples, device)

        all_states_list.append(states)
        all_ts_list.append(ts)
        fig_method_names.append(method_name)
        fig_method_colors.append(method_info['color'])

        del sde, source
        torch.cuda.empty_cache()

    print("\n  Plotting stacked marginal evolution...")
    plot_marginal_stacked(
        energy, all_states_list, all_ts_list, centers,
        fig_method_names, fig_method_colors,
        FIG_DIR / 'grid25_sdr_marginal_neurips.png',
        n_snapshots=6,
    )

    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"\n=== Evaluation complete at {now_kst} ===")


if __name__ == '__main__':
    main()
