"""
scripts/eval_2d_viz.py

Generates publication-quality 2D visualization figures:
  - Figure A: Terminal distribution (4-panel)
  - Figure B: Trajectory visualization (3-panel)

Also computes mode coverage metrics and appends to RESULTS.md.

Usage:
    python scripts/eval_2d_viz.py \
        --benchmark gmm9 \
        --baseline_dir outputs/gmm9_asbs/0 \
        --ksd_dir outputs/gmm9_ksd_asbs/0 \
        --output_dir results/figures_2d \
        --results_md RESULTS.md
"""

import argparse
import torch
import numpy as np
from pathlib import Path

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

def load_model(exp_dir, device):
    """Load trained model from experiment directory."""
    exp_dir = Path(exp_dir)

    cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / 'config.yaml'

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
# Sample and trajectory generation
# ====================================================================

@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device):
    """Generate terminal samples."""
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


@torch.no_grad()
def generate_trajectories(sde, source, ts_cfg, n_traj, device):
    """Generate full trajectories for visualization.

    Returns:
        trajectories: (n_traj, T, 2) tensor -- full paths
        x1: (n_traj, 2) -- terminal samples
    """
    x0 = source.sample([n_traj]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    states = sdeint(sde, x0, ts, only_boundary=False)
    # states is a list of T tensors, each (n_traj, D)
    trajectories = torch.stack(states, dim=1)  # (n_traj, T, D)
    x1 = states[-1]
    return trajectories, x1


# ====================================================================
# Metrics
# ====================================================================

def assign_modes(samples, centers, threshold_factor=3.0, std=0.3):
    """Assign each sample to nearest mode center.

    Returns:
        assignments: (N,) long tensor -- mode index per sample (-1 if too far)
        counts: (K,) -- samples per mode
        n_covered: int -- modes with at least 1 sample within threshold
    """
    # (N, K)
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)  # (N,)
    min_dists = dists.min(dim=1).values  # (N,)

    threshold = threshold_factor * std
    assignments = nearest.clone()
    assignments[min_dists > threshold] = -1

    K = centers.shape[0]
    counts = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        counts[k] = (assignments == k).sum()

    n_covered = (counts > 0).sum().item()
    return assignments, counts, n_covered


def compute_2d_metrics(samples, energy, centers, std):
    """Compute metrics for 2D benchmark."""
    assignments, counts, n_covered = assign_modes(
        samples, centers.to(samples.device), std=std
    )
    E = energy.eval(samples)

    return {
        'n_modes_covered': n_covered,
        'n_modes_total': centers.shape[0],
        'per_mode_counts': counts.cpu().tolist(),
        'mean_energy': E.mean().item(),
        'std_energy': E.std().item(),
    }


# ====================================================================
# Figure A: Terminal distribution (4-panel)
# ====================================================================

def plot_density_contours(ax, energy, xlim, ylim, n_grid=200):
    """Plot energy density contours on an axis."""
    x = torch.linspace(xlim[0], xlim[1], n_grid)
    y = torch.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        E = energy.eval(grid).reshape(n_grid, n_grid)

    log_p = -E
    log_p = log_p - log_p.max()
    p = torch.exp(log_p)

    ax.contourf(xx.numpy(), yy.numpy(), p.numpy(),
                levels=30, cmap='Blues', alpha=0.6)
    ax.contour(xx.numpy(), yy.numpy(), p.numpy(),
               levels=10, colors='steelblue', alpha=0.4, linewidths=0.5)


def plot_terminal_distribution(
    energy, samples_ref, samples_base, samples_ksd,
    centers, base_modes, ksd_modes,
    benchmark_name, output_path, xlim, ylim,
):
    """4-panel figure: Reference density, Baseline samples, KSD samples, Overlay."""

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    titles = [
        f'Target Density',
        f'Baseline ASBS\n({base_modes} modes covered)',
        f'SDR-ASBS\n({ksd_modes} modes covered)',
        f'Overlay (Both)',
    ]

    for ax, title in zip(axes, titles):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

        # Plot mode centers
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=100, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Panel 0: Reference samples
    r = samples_ref.cpu().numpy()
    axes[0].scatter(r[:, 0], r[:, 1], s=2, c='gray', alpha=0.3, zorder=5)

    # Panel 1: Baseline samples
    b = samples_base.cpu().numpy()
    axes[1].scatter(b[:, 0], b[:, 1], s=4, c='#d62728', alpha=0.5, zorder=5)

    # Panel 2: KSD samples
    k = samples_ksd.cpu().numpy()
    axes[2].scatter(k[:, 0], k[:, 1], s=4, c='#ff7f0e', alpha=0.5, zorder=5)

    # Panel 3: Overlay
    axes[3].scatter(b[:, 0], b[:, 1], s=3, c='#d62728', alpha=0.3, zorder=5, label='Baseline')
    axes[3].scatter(k[:, 0], k[:, 1], s=3, c='#ff7f0e', alpha=0.3, zorder=6, label='SDR-ASBS')
    axes[3].legend(fontsize=10, loc='upper right')

    fig.suptitle(f'{benchmark_name}: Terminal Distribution', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved terminal distribution: {output_path}")


# ====================================================================
# Figure B: Trajectory visualization (3-panel)
# ====================================================================

MODE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
]

def plot_trajectories(
    energy, traj_base, traj_ksd, x1_base, x1_ksd,
    centers, std, benchmark_name, output_path, xlim, ylim,
    n_show=80, stride=5,
):
    """3-panel figure: Density, Baseline trajectories, KSD trajectories.

    Each trajectory is a thin line colored by its terminal mode assignment.
    stride: plot every stride-th timestep for cleaner lines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = ['Target Density', 'Baseline ASBS Trajectories', 'SDR-ASBS Trajectories']
    for ax, title in zip(axes, titles):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=120, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Assign colors by terminal mode
    assign_base, _, _ = assign_modes(x1_base, centers.to(x1_base.device), std=std)
    assign_ksd, _, _ = assign_modes(x1_ksd, centers.to(x1_ksd.device), std=std)

    def draw_trajectories(ax, traj, assignments, n_show):
        """Draw n_show trajectories, colored by mode assignment."""
        idx = torch.randperm(traj.shape[0])[:n_show]
        for i in idx:
            path = traj[i, ::stride].cpu().numpy()  # (T//stride, 2)
            mode = assignments[i].item()
            color = MODE_COLORS[mode % len(MODE_COLORS)] if mode >= 0 else '#cccccc'
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.3, linewidth=0.6, zorder=3)
            # Terminal point
            ax.scatter(path[-1, 0], path[-1, 1], s=8, c=color, zorder=7,
                       edgecolors='black', linewidths=0.3)

    # Panel 0: reference samples (no trajectories)
    ref_samples = generate_reference_samples(energy, centers, std, n=2000)
    axes[0].scatter(ref_samples[:, 0].numpy(), ref_samples[:, 1].numpy(),
                    s=2, c='gray', alpha=0.3, zorder=5)

    # Panel 1: baseline trajectories
    draw_trajectories(axes[1], traj_base, assign_base, n_show)

    # Panel 2: KSD trajectories
    draw_trajectories(axes[2], traj_ksd, assign_ksd, n_show)

    fig.suptitle(f'{benchmark_name}: SDE Trajectories', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory plot: {output_path}")


def generate_reference_samples(energy, centers, std, n=2000):
    """Generate reference samples by sampling uniformly from all modes."""
    if hasattr(energy, 'get_ref_samples'):
        return energy.get_ref_samples(n)
    K = centers.shape[0]
    n_per = n // K
    samples = []
    for k in range(K):
        s = torch.randn(n_per, 2) * std + centers[k]
        samples.append(s)
    return torch.cat(samples, dim=0)


# ====================================================================
# Main
# ====================================================================

BENCHMARK_CONFIGS = {
    'gmm9': {
        'xlim': (-8, 8), 'ylim': (-8, 8),
        'name': '9-Mode GMM (3x3 Grid)',
    },
    'ring8': {
        'xlim': (-8, 8), 'ylim': (-8, 8),
        'name': '8-Mode Ring',
    },
    'banana': {
        'xlim': (-4, 4), 'ylim': (-2, 5),
        'name': 'Double Banana (70/30)',
        'threshold_factor': 5.0,  # Banana modes need generous threshold
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['gmm9', 'ring8', 'banana'])
    parser.add_argument('--baseline_dir', type=str, required=True)
    parser.add_argument('--ksd_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/figures_2d')
    parser.add_argument('--results_md', type=str, default=None,
                        help='Path to RESULTS.md to append results')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_traj', type=int, default=200)
    parser.add_argument('--n_traj_show', type=int, default=80)
    parser.add_argument('--traj_stride', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_cfg = BENCHMARK_CONFIGS[args.benchmark]
    threshold_factor = bench_cfg.get('threshold_factor', 3.0)

    print(f"=== 2D Visualization: {bench_cfg['name']} ===")

    # Load models
    print("Loading baseline...")
    sde_base, src_base, energy, ts_base = load_model(args.baseline_dir, device)
    print("Loading KSD-ASBS...")
    sde_ksd, src_ksd, _, ts_ksd = load_model(args.ksd_dir, device)

    centers = energy.get_centers().to(device)
    std = energy.get_std()
    K = centers.shape[0]

    # --- Generate terminal samples ---
    print(f"Generating {args.n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, args.n_samples, device)
    torch.manual_seed(0)
    samples_ksd = generate_samples(sde_ksd, src_ksd, ts_ksd, args.n_samples, device)
    samples_ref = generate_reference_samples(energy, centers, std, args.n_samples).to(device)

    # --- Metrics ---
    print("Computing metrics...")
    m_base = compute_2d_metrics(samples_base, energy, centers, std)
    m_ksd = compute_2d_metrics(samples_ksd, energy, centers, std)

    print(f"  Baseline: {m_base['n_modes_covered']}/{K} modes, "
          f"mean_E={m_base['mean_energy']:.3f}")
    print(f"  KSD-ASBS: {m_ksd['n_modes_covered']}/{K} modes, "
          f"mean_E={m_ksd['mean_energy']:.3f}")
    print(f"  Baseline per-mode: {m_base['per_mode_counts']}")
    print(f"  KSD-ASBS per-mode: {m_ksd['per_mode_counts']}")

    # --- Figure A: Terminal distribution ---
    print("Generating terminal distribution figure...")
    plot_terminal_distribution(
        energy, samples_ref, samples_base, samples_ksd,
        centers, m_base['n_modes_covered'], m_ksd['n_modes_covered'],
        bench_cfg['name'],
        output_dir / f'{args.benchmark}_terminal.png',
        bench_cfg['xlim'], bench_cfg['ylim'],
    )

    # --- Generate trajectories ---
    print(f"Generating {args.n_traj} trajectories...")
    torch.manual_seed(42)
    traj_base, x1_base = generate_trajectories(
        sde_base, src_base, ts_base, args.n_traj, device
    )
    torch.manual_seed(42)
    traj_ksd, x1_ksd = generate_trajectories(
        sde_ksd, src_ksd, ts_ksd, args.n_traj, device
    )

    # --- Figure B: Trajectory visualization ---
    print("Generating trajectory figure...")
    plot_trajectories(
        energy, traj_base, traj_ksd, x1_base, x1_ksd,
        centers, std, bench_cfg['name'],
        output_dir / f'{args.benchmark}_trajectories.png',
        bench_cfg['xlim'], bench_cfg['ylim'],
        n_show=args.n_traj_show, stride=args.traj_stride,
    )

    # --- Append to RESULTS.md ---
    if args.results_md:
        append_results(args.results_md, args.benchmark, bench_cfg,
                       m_base, m_ksd, output_dir)

    print(f"=== Done: {bench_cfg['name']} ===\n")


def append_results(results_md_path, benchmark, bench_cfg, m_base, m_ksd, fig_dir):
    """Append results for this benchmark to RESULTS.md."""
    K = m_base['n_modes_total']

    section = f"""

---

### 2D Visualization: {bench_cfg['name']}

| Metric | Baseline ASBS | SDR-ASBS | Delta |
|---|---|---|---|
| Modes covered (of {K}) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} | {m_ksd['n_modes_covered'] - m_base['n_modes_covered']:+d} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} | {(m_ksd['mean_energy'] - m_base['mean_energy']) / (abs(m_base['mean_energy']) + 1e-10) * 100:+.1f}% |
| Per-mode counts (base) | {m_base['per_mode_counts']} | | |
| Per-mode counts (SDR) | | {m_ksd['per_mode_counts']} | |

**Terminal Distribution:**

![{benchmark} terminal]({fig_dir}/{benchmark}_terminal.png)

**SDE Trajectories** -- Baseline trajectories converge to {m_base['n_modes_covered']} mode(s); SDR trajectories fan out to {m_ksd['n_modes_covered']} mode(s). Each line is one SDE trajectory from source to terminal time, colored by terminal mode assignment.

![{benchmark} trajectories]({fig_dir}/{benchmark}_trajectories.png)

"""
    with open(results_md_path, 'a') as f:
        f.write(section)
    print(f"  Appended results to {results_md_path}")


if __name__ == '__main__':
    main()
