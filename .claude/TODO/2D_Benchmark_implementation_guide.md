# 2D Visualization Benchmarks: Implementation Guide

## For Claude Code — New Files Only, No Modifications to Existing Code

-----

## 1. Purpose

Add three 2D energy functions to produce publication-quality figures showing:

- **Terminal distribution**: samples overlaid on density contours (4-panel: Reference, Baseline, SDR-ASBS, overlay)
- **Trajectory visualization**: SDE paths from source to target, colored by terminal mode (3-panel: Reference, Baseline trajectories, SDR trajectories)

These figures directly visualize the mode coverage mechanism of SDR. The trajectory plot is unique to SOC-based methods and is the strongest visual in the paper.

-----

## 2. New Files

```
adjoint_samplers/
├── energies/
│   └── viz_energies.py              # NEW — all three 2D energy functions
├── components/
│   └── (no changes)
├── configs/
│   ├── problem/
│   │   ├── gmm9.yaml                # NEW
│   │   ├── ring8.yaml               # NEW
│   │   └── banana.yaml              # NEW
│   └── experiment/
│       ├── gmm9_asbs.yaml           # NEW
│       ├── gmm9_ksd_asbs.yaml       # NEW
│       ├── ring8_asbs.yaml          # NEW
│       ├── ring8_ksd_asbs.yaml      # NEW
│       ├── banana_asbs.yaml         # NEW
│       └── banana_ksd_asbs.yaml     # NEW
├── scripts/
│   └── eval_2d_viz.py               # NEW — evaluation + figure generation
```

-----

## 3. Energy Functions (`adjoint_samplers/energies/viz_energies.py`)

```python
"""
adjoint_samplers/energies/viz_energies.py

Three 2D energy functions for visualization benchmarks.
All have known mode structure for quantitative evaluation.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class GMM9Energy(BaseEnergy):
    """9-mode Gaussian mixture on a 3x3 grid.

    From DGFS (Zhang et al., ICLR 2024).
    Modes at {-5, 0, 5} x {-5, 0, 5}, each with σ=0.3.

    p(x) ∝ Σ_{k=1}^{9} N(x; μ_k, 0.3² I)
    """
    def __init__(self, dim=2, device="cpu"):
        super().__init__("gmm9", dim)
        assert dim == 2

        centers = []
        for i in [-5.0, 0.0, 5.0]:
            for j in [-5.0, 0.0, 5.0]:
                centers.append([i, j])
        self.centers = torch.tensor(centers, dtype=torch.float32)  # (9, 2)
        self.std = 0.3
        self.n_modes = 9

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) → E: (B,)"""
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        # (B, 9, 2)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, 9)
        log_probs = -sq_dist / (2 * self.std ** 2)
        return -torch.logsumexp(log_probs, dim=1)

    def get_centers(self):
        return self.centers

    def get_std(self):
        return self.std


class Ring8Energy(BaseEnergy):
    """8 Gaussians on a ring.

    2D version of the RotGMM benchmark. Modes at radius r=5,
    equally spaced angles. σ=0.3 per mode.
    """
    def __init__(self, dim=2, radius=5.0, std=0.3, n_modes=8, device="cpu"):
        super().__init__("ring8", dim)
        assert dim == 2
        self.std = std
        self.n_modes = n_modes
        self.radius = radius

        angles = torch.linspace(0, 2 * np.pi, n_modes + 1)[:-1]
        centers = torch.stack([radius * torch.cos(angles),
                               radius * torch.sin(angles)], dim=1)  # (8, 2)
        self.centers = centers

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)
        log_probs = -sq_dist / (2 * self.std ** 2)
        return -torch.logsumexp(log_probs, dim=1)

    def get_centers(self):
        return self.centers

    def get_std(self):
        return self.std


class BananaEnergy(BaseEnergy):
    """Unequal-weight double banana potential.

    Two banana-shaped modes with 70/30 weight split.
    Mode-seeking methods collapse to the 70% mode.

    E(x) = -log[0.7 * exp(-(x2 - x1²)²/0.2 - (x1 - 1)²/2)
              + 0.3 * exp(-(x2 - x1²)²/0.2 - (x1 + 1)²/2)]
    """
    def __init__(self, dim=2, device="cpu"):
        super().__init__("banana", dim)
        assert dim == 2
        self.n_modes = 2
        self.weights = [0.7, 0.3]

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]

        banana_term = (x2 - x1 ** 2) ** 2 / 0.2

        log_mode1 = -banana_term - (x1 - 1) ** 2 / 2 + np.log(0.7)
        log_mode2 = -banana_term - (x1 + 1) ** 2 / 2 + np.log(0.3)

        log_p = torch.logsumexp(torch.stack([log_mode1, log_mode2], dim=-1), dim=-1)
        return -log_p

    def get_centers(self):
        """Approximate mode centers (peaks of banana curves)."""
        return torch.tensor([[1.0, 1.0], [-1.0, 1.0]])

    def get_std(self):
        return 0.5  # approximate spread
```

-----

## 4. Config Files

### Problem Configs

`configs/problem/gmm9.yaml`:

```yaml
# @package _global_
dim: 2
n_particles: 1
spatial_dim: 2

energy:
  _target_: adjoint_samplers.energies.viz_energies.GMM9Energy
  dim: ${dim}
```

`configs/problem/ring8.yaml`:

```yaml
# @package _global_
dim: 2
n_particles: 1
spatial_dim: 2

energy:
  _target_: adjoint_samplers.energies.viz_energies.Ring8Energy
  dim: ${dim}
  radius: 5.0
  std: 0.3
  n_modes: 8
```

`configs/problem/banana.yaml`:

```yaml
# @package _global_
dim: 2
n_particles: 1
spatial_dim: 2

energy:
  _target_: adjoint_samplers.energies.viz_energies.BananaEnergy
  dim: ${dim}
```

### Experiment Configs

All three share the same training setup. Only the problem and exp_name differ.

`configs/experiment/gmm9_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: gmm9
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: term_cost

exp_name: gmm9_asbs
nfe: 100
sigma_max: 8
sigma_min: 0.01
rescale_t: null
num_epochs: 3000
max_grad_E_norm: 50

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/gmm9_ksd_asbs.yaml`: Same but change matcher to `ksd_adjoint_ve` and add `sdr_lambda: 1.0`.

`configs/experiment/ring8_asbs.yaml`: Same as gmm9_asbs but `exp_name: ring8_asbs` and problem `ring8`.

`configs/experiment/ring8_ksd_asbs.yaml`: Same with SDR matcher and `sdr_lambda: 1.0`.

`configs/experiment/banana_asbs.yaml`: Same as gmm9_asbs but `exp_name: banana_asbs`, problem `banana`, and `sigma_max: 3` (banana modes are closer together).

`configs/experiment/banana_ksd_asbs.yaml`: Same with SDR matcher and `sdr_lambda: 0.5` (start lower — banana has sharper gradients than GMM).

-----

## 5. Training

```bash
# ~10 minutes each, all seed=0
for BENCH in gmm9 ring8 banana; do
  python train.py experiment=${BENCH}_asbs seed=0 use_wandb=false
  python train.py experiment=${BENCH}_ksd_asbs seed=0 use_wandb=false
done
```

-----

## 6. Evaluation and Figure Generation (`scripts/eval_2d_viz.py`)

This is the main script. It loads trained checkpoints, generates samples and trajectories, computes metrics, produces figures, and appends results to RESULTS.md.

```python
"""
scripts/eval_2d_viz.py

Generates publication-quality 2D visualization figures:
  - Figure A: Terminal distribution (4-panel)
  - Figure B: Trajectory visualization (3-panel)

Also computes mode coverage metrics and appends to RESULTS.md.

Usage:
    python scripts/eval_2d_viz.py \
        --benchmark gmm9 \
        --baseline_dir outputs/gmm9_asbs \
        --ksd_dir outputs/gmm9_ksd_asbs \
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
from matplotlib.colors import LogNorm

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
        't0': cfg.timesteps.t0, 't1': cfg.timesteps.t1,
        'steps': cfg.timesteps.steps, 'rescale_t': cfg.timesteps.rescale_t,
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
        trajectories: (n_traj, T, 2) tensor — full paths
        x1: (n_traj, 2) — terminal samples
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
        assignments: (N,) long tensor — mode index per sample (-1 if too far)
        counts: (K,) — samples per mode
        n_covered: int — modes with at least 1 sample within threshold
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
    assignments, counts, n_covered = assign_modes(samples, centers.to(samples.device), std=std)
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
    """4-panel figure: Reference density, Baseline samples, SDR samples, Overlay."""

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
        ax.scatter(c[:, 0], c[:, 1], marker='☆', s=100, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Panel 0: Reference samples
    r = samples_ref.cpu().numpy()
    axes[0].scatter(r[:, 0], r[:, 1], s=2, c='gray', alpha=0.3, zorder=5)

    # Panel 1: Baseline samples
    b = samples_base.cpu().numpy()
    axes[1].scatter(b[:, 0], b[:, 1], s=4, c='#d62728', alpha=0.5, zorder=5)

    # Panel 2: SDR samples
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
    """3-panel figure: Density, Baseline trajectories, SDR trajectories.

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
        ax.scatter(c[:, 0], c[:, 1], marker='☆', s=120, c='black',
                   zorder=10, edgecolors='white', linewidths=0.5)

    # Assign colors by terminal mode
    assign_base, _, _ = assign_modes(x1_base, centers.to(x1_base.device), std=std)
    assign_ksd, _, _ = assign_modes(x1_ksd, centers.to(x1_ksd.device), std=std)

    K = centers.shape[0]

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

    # Panel 2: SDR trajectories
    draw_trajectories(axes[2], traj_ksd, assign_ksd, n_show)

    fig.suptitle(f'{benchmark_name}: SDE Trajectories', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory plot: {output_path}")


def generate_reference_samples(energy, centers, std, n=2000):
    """Generate reference samples by sampling uniformly from all modes."""
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
        'name': '9-Mode GMM (3×3 Grid)',
    },
    'ring8': {
        'xlim': (-8, 8), 'ylim': (-8, 8),
        'name': '8-Mode Ring',
    },
    'banana': {
        'xlim': (-4, 4), 'ylim': (-2, 5),
        'name': 'Double Banana (70/30)',
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

    print(f"=== 2D Visualization: {bench_cfg['name']} ===")

    # Load models
    print("Loading baseline...")
    sde_base, src_base, energy, ts_base = load_model(args.baseline_dir, device)
    print("Loading SDR-ASBS...")
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
    print(f"  SDR-ASBS: {m_ksd['n_modes_covered']}/{K} modes, "
          f"mean_E={m_ksd['mean_energy']:.3f}")
    print(f"  Baseline per-mode: {m_base['per_mode_counts']}")
    print(f"  SDR-ASBS per-mode: {m_ksd['per_mode_counts']}")

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

| Metric | Baseline ASBS | SDR-ASBS | Δ |
|---|---|---|---|
| Modes covered (of {K}) | {m_base['n_modes_covered']} | {m_ksd['n_modes_covered']} | {m_ksd['n_modes_covered'] - m_base['n_modes_covered']:+d} |
| Mean energy | {m_base['mean_energy']:.4f} | {m_ksd['mean_energy']:.4f} | {(m_ksd['mean_energy'] - m_base['mean_energy']) / (abs(m_base['mean_energy']) + 1e-10) * 100:+.1f}% |
| Per-mode counts (base) | {m_base['per_mode_counts']} | | |
| Per-mode counts (SDR) | | {m_ksd['per_mode_counts']} | |

**Terminal Distribution:**

![{benchmark} terminal]({fig_dir}/{benchmark}_terminal.png)

**SDE Trajectories** — Baseline trajectories converge to {m_base['n_modes_covered']} mode(s); SDR trajectories fan out to {m_ksd['n_modes_covered']} mode(s). Each line is one SDE trajectory from source to terminal time, colored by terminal mode assignment.

![{benchmark} trajectories]({fig_dir}/{benchmark}_trajectories.png)

"""
    with open(results_md_path, 'a') as f:
        f.write(section)
    print(f"  Appended results to {results_md_path}")


if __name__ == '__main__':
    main()
```

-----

## 7. Running the Full Pipeline

```bash
# Train (all seed=0, ~10 min each)
for BENCH in gmm9 ring8 banana; do
  python train.py experiment=${BENCH}_asbs seed=0 use_wandb=false
  python train.py experiment=${BENCH}_ksd_asbs seed=0 use_wandb=false
done

# Evaluate + generate figures
for BENCH in gmm9 ring8 banana; do
  python scripts/eval_2d_viz.py \
    --benchmark ${BENCH} \
    --baseline_dir outputs/${BENCH}_asbs \
    --ksd_dir outputs/${BENCH}_ksd_asbs \
    --output_dir results/figures_2d \
    --results_md RESULTS.md \
    --n_samples 2000 \
    --n_traj 200
done
```

-----

## 8. Expected Outputs

### Figures Generated (6 total)

```
results/figures_2d/
├── gmm9_terminal.png       # 4-panel: density, baseline, SDR, overlay
├── gmm9_trajectories.png   # 3-panel: density, baseline trajs, SDR trajs
├── ring8_terminal.png
├── ring8_trajectories.png
├── banana_terminal.png
└── banana_trajectories.png
```

### What to Expect Visually

**GMM9 (3×3 grid):**

- Baseline: samples cluster at 3-5 central modes, corners missing
- SDR-ASBS: samples cover 7-9 modes including corners
- Trajectories: baseline paths converge to center of grid; SDR paths fan out to corners

**Ring8 (ring):**

- Baseline: samples collapse to 1-2 adjacent modes on the ring
- SDR-ASBS: samples spread across 4-8 modes around the ring
- Trajectories: this is the 2D version of your RotGMM d=10 result — directly comparable

**Banana (70/30 split):**

- Baseline: all samples in the 70% banana (right), minority banana (left) empty
- SDR-ASBS: samples in both bananas, roughly 70/30 split matching the true weights
- Trajectories: baseline paths all curve right; SDR paths split left and right

### What to Expect in Metrics

|Benchmark|Baseline modes|SDR modes (expected)|
|---------|--------------|--------------------|
|GMM9     |3–5 of 9      |7–9 of 9            |
|Ring8    |1–2 of 8      |3–6 of 8            |
|Banana   |1 of 2        |2 of 2              |

-----

## 10. Additional 2D Benchmarks (Phase 2)

These benchmarks go beyond separated Gaussians to test SDR on harder geometry.

### 10.1 Two Moons

Two crescent-shaped modes that wrap around each other. Unlike Gaussians, the modes are non-convex and interleaved. Tests whether SDR can handle non-Gaussian geometry, not just "find separated blobs."

- **Modes**: 2 crescent-shaped (non-convex)
- **Challenge**: Interleaved non-convex geometry — modes are not separable by any hyperplane
- **sigma_max**: 3 (modes are close, radius ~1-2)
- **Mode coverage metric**: Assign by sign of a learned or geometric boundary, or by nearest centroid of each crescent

### 10.2 Pinwheel

5 elongated clusters arranged in a pinwheel/spiral pattern. Each arm is a thin stretched Gaussian rotated at a different angle. Mode-seeking methods collapse to the thickest arm.

- **Modes**: 5 anisotropic (elongated, rotated)
- **Challenge**: Anisotropic mode geometry — modes have very different aspect ratios and orientations
- **sigma_max**: 5 (arms extend to radius ~3-4)
- **Mode coverage metric**: Nearest-centroid assignment to arm centers

### 10.3 Checkerboard

4x4 grid of alternating high/low density squares. Not separated Gaussians — the modes share boundaries. Harder than GMM9 because there's no empty space between modes, so the sampler must learn sharp density boundaries.

- **Modes**: 8 (the "on" squares in 4x4 checkerboard)
- **Challenge**: Shared boundaries between modes — no gaps. Sharp density transitions.
- **sigma_max**: 5 (grid spans ~[-4, 4])
- **Mode coverage metric**: Count how many of the 8 "on" squares have samples

### 10.4 Nested Rings

Two concentric rings at different radii (e.g., r=2 and r=5) with different weights (80/20). The inner ring is hard to find because the outer ring surrounds it.

- **Modes**: 2 (inner ring, outer ring)
- **Challenge**: Inner mode is geometrically surrounded/shielded by outer mode. Tests whether SDR can push samples inward, not just outward.
- **sigma_max**: 8 (outer ring at r=5)
- **Mode coverage metric**: Fraction of samples within radius threshold of each ring

### 10.5 Spiral

A single continuous spiral-shaped density. Not multimodal, but tests whether the SDE can learn complex non-convex geometry. Good contrast with the multimodal benchmarks — shows the method doesn't hurt on unimodal targets.

- **Modes**: 1 (continuous spiral)
- **Challenge**: Complex non-convex unimodal geometry. No mode collapse possible, but requires learning curved transport.
- **sigma_max**: 5 (spiral extends to radius ~4)
- **Mode coverage metric**: Not applicable (unimodal). Use energy W2 or sample quality metrics instead.

### Summary Table

| Benchmark | Modes | Geometry | Key Test |
|-----------|-------|----------|----------|
| GMM9 (done) | 9 Gaussians | Convex, separated | Basic mode coverage |
| Two Moons | 2 crescents | Non-convex, interleaved | Non-Gaussian geometry |
| Pinwheel | 5 elongated | Anisotropic, rotated | Anisotropic modes |
| Checkerboard | 8 squares | Shared boundaries | Sharp density transitions |
| Nested Rings | 2 rings | Concentric, shielded | Inward exploration |
| Spiral | 1 spiral | Non-convex, unimodal | Complex transport (control) |

-----

## 9. Notes for Claude Code

1. **`sdeint` with `only_boundary=False`** returns a list of tensors, one per timestep. Stack with `torch.stack(states, dim=1)` to get `(B, T, D)`.
1. **Same seed for baseline and SDR** when generating trajectories (`torch.manual_seed(42)`). This means both models start from the same initial positions — the difference in trajectories is purely due to the controller, making the comparison fair and visually striking.
1. **`energy.get_centers()` and `energy.get_std()`** are custom methods on each energy class. Make sure they return CPU tensors.
1. **The `☆` marker** in matplotlib requires unicode support. If it fails, replace with `'*'` (star marker).
1. **Banana energy** — the mode centers from `get_centers()` are approximate (the banana peak is not exactly at (1,1) and (-1,1)). Use nearest-mode assignment with a generous threshold (e.g., `threshold_factor=5.0`).
1. **Do NOT modify** `sde.py`, `matcher.py`, `sdr_matcher.py`, `stein_kernel.py`, `train.py`, `train_loop.py`, or any existing config files. All new code goes in new files.
1. **Figure resolution**: `dpi=200` for paper quality. The trajectory figure with 80 paths and stride=5 should render cleanly. If paths are too dense, reduce `n_traj_show` to 50.
1. **sigma_max for GMM9 and Ring8**: Set to 8 (modes are at distance 5–7 from origin). For Banana: set to 3 (modes are at distance ~1–2 from origin). Too-small sigma_max means the noise can’t reach the modes; too-large makes training harder.