"""
scripts/eval_2d_all.py

Evaluates all 6 2D experiments (3 per benchmark: baseline, ksd_warmup, ksd_nowarmup)
and generates:
  - 6-panel terminal distribution figure per benchmark
  - 4-panel trajectory figure per benchmark
  - 2d_result.md with all figures and metrics
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

def load_model(exp_dir, device):
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
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


@torch.no_grad()
def generate_trajectories(sde, source, ts_cfg, n_traj, device):
    x0 = source.sample([n_traj]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    states = sdeint(sde, x0, ts, only_boundary=False)
    trajectories = torch.stack(states, dim=1)
    x1 = states[-1]
    return trajectories, x1


def generate_reference_samples(energy, centers, std, n=2000):
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


def compute_2d_metrics(samples, energy, centers, std):
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
# Plotting helpers
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


MODE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
]


# ====================================================================
# Figure: Terminal distribution (6-panel for 3-way comparison)
# ====================================================================

def plot_terminal_3way(
    energy, samples_ref, samples_base, samples_ksd_warm, samples_ksd_nowarm,
    centers, m_base, m_ksd_warm, m_ksd_nowarm,
    benchmark_name, output_path, xlim, ylim,
):
    """6-panel: Reference | Baseline | KSD-Warmup | KSD-NoWarmup | Overlay(Base vs Warm) | Overlay(Warm vs NoWarm)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    titles = [
        'Target Density',
        f'Baseline ASBS\n({m_base["n_modes_covered"]} modes)',
        f'KSD-Warmup\n({m_ksd_warm["n_modes_covered"]} modes)',
        f'KSD-NoWarmup\n({m_ksd_nowarm["n_modes_covered"]} modes)',
        'Overlay: Baseline vs KSD-Warmup',
        'Overlay: KSD-Warmup vs KSD-NoWarmup',
    ]

    for ax, title in zip(axes, titles):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=80, c='black', zorder=10, edgecolors='white', linewidths=0.5)

    # Panel 0: Reference
    r = samples_ref.cpu().numpy()
    axes[0].scatter(r[:, 0], r[:, 1], s=2, c='gray', alpha=0.3, zorder=5)

    # Panel 1: Baseline
    b = samples_base.cpu().numpy()
    axes[1].scatter(b[:, 0], b[:, 1], s=4, c='#d62728', alpha=0.5, zorder=5)

    # Panel 2: KSD-Warmup
    kw = samples_ksd_warm.cpu().numpy()
    axes[2].scatter(kw[:, 0], kw[:, 1], s=4, c='#ff7f0e', alpha=0.5, zorder=5)

    # Panel 3: KSD-NoWarmup
    knw = samples_ksd_nowarm.cpu().numpy()
    axes[3].scatter(knw[:, 0], knw[:, 1], s=4, c='#2ca02c', alpha=0.5, zorder=5)

    # Panel 4: Overlay Base vs Warmup
    axes[4].scatter(b[:, 0], b[:, 1], s=3, c='#d62728', alpha=0.3, zorder=5, label='Baseline')
    axes[4].scatter(kw[:, 0], kw[:, 1], s=3, c='#ff7f0e', alpha=0.3, zorder=6, label='KSD-Warmup')
    axes[4].legend(fontsize=9, loc='upper right')

    # Panel 5: Overlay Warmup vs NoWarmup
    axes[5].scatter(kw[:, 0], kw[:, 1], s=3, c='#ff7f0e', alpha=0.3, zorder=5, label='KSD-Warmup')
    axes[5].scatter(knw[:, 0], knw[:, 1], s=3, c='#2ca02c', alpha=0.3, zorder=6, label='KSD-NoWarmup')
    axes[5].legend(fontsize=9, loc='upper right')

    fig.suptitle(f'{benchmark_name}: Terminal Distribution (3-way)', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ====================================================================
# Figure: Trajectory visualization (4-panel for 3-way)
# ====================================================================

def plot_trajectories_3way(
    energy, traj_base, traj_ksd_warm, traj_ksd_nowarm,
    x1_base, x1_ksd_warm, x1_ksd_nowarm,
    centers, std, benchmark_name, output_path, xlim, ylim,
    n_show=80, stride=5,
):
    """4-panel: Density | Baseline trajs | KSD-Warmup trajs | KSD-NoWarmup trajs"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    titles = ['Target Density', 'Baseline Trajectories', 'KSD-Warmup Trajectories', 'KSD-NoWarmup Trajectories']
    for ax, title in zip(axes, titles):
        plot_density_contours(ax, energy, xlim, ylim)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        c = centers.cpu().numpy()
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=100, c='black', zorder=10, edgecolors='white', linewidths=0.5)

    assign_base, _, _ = assign_modes(x1_base, centers.to(x1_base.device), std=std)
    assign_warm, _, _ = assign_modes(x1_ksd_warm, centers.to(x1_ksd_warm.device), std=std)
    assign_nowarm, _, _ = assign_modes(x1_ksd_nowarm, centers.to(x1_ksd_nowarm.device), std=std)

    def draw_trajectories(ax, traj, assignments, n_show):
        idx = torch.randperm(traj.shape[0])[:n_show]
        for i in idx:
            path = traj[i, ::stride].cpu().numpy()
            mode = assignments[i].item()
            color = MODE_COLORS[mode % len(MODE_COLORS)] if mode >= 0 else '#cccccc'
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.3, linewidth=0.6, zorder=3)
            ax.scatter(path[-1, 0], path[-1, 1], s=8, c=color, zorder=7, edgecolors='black', linewidths=0.3)

    # Panel 0: reference samples
    ref_samples = generate_reference_samples(energy, centers, std, n=2000)
    axes[0].scatter(ref_samples[:, 0].numpy(), ref_samples[:, 1].numpy(), s=2, c='gray', alpha=0.3, zorder=5)

    draw_trajectories(axes[1], traj_base, assign_base, n_show)
    draw_trajectories(axes[2], traj_ksd_warm, assign_warm, n_show)
    draw_trajectories(axes[3], traj_ksd_nowarm, assign_nowarm, n_show)

    fig.suptitle(f'{benchmark_name}: SDE Trajectories (3-way)', fontsize=15, y=1.02)
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

BENCHMARKS = {
    'gmm9': {
        'name': '9-Mode GMM (3x3 Grid)',
        'xlim': (-8, 8), 'ylim': (-8, 8),
        'baseline': RESULTS_DIR / 'gmm9_asbs' / 'seed_0',
        'ksd_warmup': RESULTS_DIR / 'gmm9_ksd_asbs' / 'seed_0',
        'ksd_nowarmup': RESULTS_DIR / 'gmm9_ksd_nowarmup' / 'seed_0',
        'ksd_lambda_warmup': 0.01,
        'ksd_lambda_nowarmup': 0.01,
    },
    'ring8': {
        'name': '8-Mode Ring',
        'xlim': (-8, 8), 'ylim': (-8, 8),
        'baseline': RESULTS_DIR / 'ring8_asbs' / 'seed_0',
        'ksd_warmup': RESULTS_DIR / 'ring8_ksd_asbs' / 'seed_0',
        'ksd_nowarmup': RESULTS_DIR / 'ring8_ksd_nowarmup' / 'seed_0',
        'ksd_lambda_warmup': 0.1,
        'ksd_lambda_nowarmup': 0.1,
    },
}


def evaluate_benchmark(bench_key, bench_cfg, device, n_samples=2000, n_traj=200, n_traj_show=80, traj_stride=5):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {bench_cfg['name']}")
    print(f"{'='*60}")

    # Load models
    print("  Loading baseline...")
    sde_base, src_base, energy, ts_base = load_model(bench_cfg['baseline'], device)
    print("  Loading KSD-Warmup...")
    sde_warm, src_warm, _, ts_warm = load_model(bench_cfg['ksd_warmup'], device)
    print("  Loading KSD-NoWarmup...")
    sde_nowarm, src_nowarm, _, ts_nowarm = load_model(bench_cfg['ksd_nowarmup'], device)

    centers = energy.get_centers().to(device)
    std = energy.get_std()

    # Generate terminal samples (same seed for fair comparison)
    print(f"  Generating {n_samples} terminal samples...")
    torch.manual_seed(0)
    samples_base = generate_samples(sde_base, src_base, ts_base, n_samples, device)
    torch.manual_seed(0)
    samples_warm = generate_samples(sde_warm, src_warm, ts_warm, n_samples, device)
    torch.manual_seed(0)
    samples_nowarm = generate_samples(sde_nowarm, src_nowarm, ts_nowarm, n_samples, device)
    samples_ref = generate_reference_samples(energy, centers, std, n_samples).to(device)

    # Metrics
    print("  Computing metrics...")
    m_base = compute_2d_metrics(samples_base, energy, centers, std)
    m_warm = compute_2d_metrics(samples_warm, energy, centers, std)
    m_nowarm = compute_2d_metrics(samples_nowarm, energy, centers, std)

    K = centers.shape[0]
    print(f"    Baseline:       {m_base['n_modes_covered']}/{K} modes, mean_E={m_base['mean_energy']:.3f}")
    print(f"    KSD-Warmup:     {m_warm['n_modes_covered']}/{K} modes, mean_E={m_warm['mean_energy']:.3f}")
    print(f"    KSD-NoWarmup:   {m_nowarm['n_modes_covered']}/{K} modes, mean_E={m_nowarm['mean_energy']:.3f}")

    # Terminal distribution figure
    print("  Generating terminal distribution figure...")
    plot_terminal_3way(
        energy, samples_ref, samples_base, samples_warm, samples_nowarm,
        centers, m_base, m_warm, m_nowarm,
        bench_cfg['name'],
        FIG_DIR / f'{bench_key}_terminal.png',
        bench_cfg['xlim'], bench_cfg['ylim'],
    )

    # Trajectories
    print(f"  Generating {n_traj} trajectories...")
    torch.manual_seed(42)
    traj_base, x1_base = generate_trajectories(sde_base, src_base, ts_base, n_traj, device)
    torch.manual_seed(42)
    traj_warm, x1_warm = generate_trajectories(sde_warm, src_warm, ts_warm, n_traj, device)
    torch.manual_seed(42)
    traj_nowarm, x1_nowarm = generate_trajectories(sde_nowarm, src_nowarm, ts_nowarm, n_traj, device)

    print("  Generating trajectory figure...")
    plot_trajectories_3way(
        energy, traj_base, traj_warm, traj_nowarm,
        x1_base, x1_warm, x1_nowarm,
        centers, std, bench_cfg['name'],
        FIG_DIR / f'{bench_key}_trajectories.png',
        bench_cfg['xlim'], bench_cfg['ylim'],
        n_show=n_traj_show, stride=traj_stride,
    )

    return {
        'name': bench_cfg['name'],
        'K': K,
        'ksd_lambda_warmup': bench_cfg['ksd_lambda_warmup'],
        'ksd_lambda_nowarmup': bench_cfg['ksd_lambda_nowarmup'],
        'm_base': m_base,
        'm_warm': m_warm,
        'm_nowarm': m_nowarm,
    }


def write_results_md(all_results):
    kst = datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

    md = f"""# 2D Visualization Benchmark Results

Generated: {kst}

---

"""
    for bench_key, res in all_results.items():
        K = res['K']
        m_b = res['m_base']
        m_w = res['m_warm']
        m_nw = res['m_nowarm']

        md += f"""## {res['name']}

| Metric | Baseline ASBS | KSD-Warmup (lambda={res['ksd_lambda_warmup']}) | KSD-NoWarmup (lambda={res['ksd_lambda_nowarmup']}) |
|---|---|---|---|
| Modes covered (of {K}) | {m_b['n_modes_covered']} | {m_w['n_modes_covered']} | {m_nw['n_modes_covered']} |
| Mean energy | {m_b['mean_energy']:.4f} | {m_w['mean_energy']:.4f} | {m_nw['mean_energy']:.4f} |
| Std energy | {m_b['std_energy']:.4f} | {m_w['std_energy']:.4f} | {m_nw['std_energy']:.4f} |
| Per-mode counts | {m_b['per_mode_counts']} | {m_w['per_mode_counts']} | {m_nw['per_mode_counts']} |

### Terminal Distribution

![{bench_key} terminal](figures_2d/{bench_key}_terminal.png)

### SDE Trajectories

![{bench_key} trajectories](figures_2d/{bench_key}_trajectories.png)

---

"""
    md_path = EVAL_DIR / '2d_result.md'
    with open(md_path, 'w') as f:
        f.write(md)
    print(f"\nWrote results to {md_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bench_key, bench_cfg in BENCHMARKS.items():
        res = evaluate_benchmark(bench_key, bench_cfg, device)
        all_results[bench_key] = res

    write_results_md(all_results)
    print("\n=== All evaluations complete! ===")


if __name__ == '__main__':
    main()
