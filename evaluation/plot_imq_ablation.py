"""
plot_imq_ablation.py — Generate 3-way comparison figures for all RotGMM dimensions.
For each dimension: mode occupation bar chart + PCA scatter (4 panels: ref, baseline, RBF-KSD, IMQ-KSD).
"""

import sys
sys.path.insert(0, "/home/RESEARCH/Stein_ASBS")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

import hydra
from omegaconf import OmegaConf
from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.utils.train_utils import get_timesteps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 2000
SEED = 0
FIG_DIR = Path("/home/RESEARCH/Stein_ASBS/evaluation/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [10, 30, 50, 100]

RUNS = {}
for d in DIMS:
    RUNS[d] = {
        "baseline": Path(f"/home/RESEARCH/Stein_ASBS/results/rotgmm{d}_asbs/seed_0"),
        "rbf_ksd": Path(f"/home/RESEARCH/Stein_ASBS/results/rotgmm{d}_ksd_asbs/seed_0"),
        "imq_ksd": Path(f"/home/RESEARCH/Stein_ASBS/results/rotgmm{d}_imq_asbs/seed_0"),
    }


def load_and_generate(run_dir, seed, n_samples):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    energy = hydra.utils.instantiate(cfg.energy, device=DEVICE)
    source = hydra.utils.instantiate(cfg.source, device=DEVICE)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(DEVICE)
    controller = hydra.utils.instantiate(cfg.controller).to(DEVICE)
    sde = ControlledSDE(ref_sde, controller).to(DEVICE)

    ckpt = torch.load(run_dir / "checkpoints" / "checkpoint_latest.pt",
                      map_location=DEVICE, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    controller.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    ts_cfg = {"t0": 0.0, "t1": 1.0,
              "steps": cfg.get("nfe", 200),
              "rescale_t": cfg.get("rescale_t", None)}

    x1_list = []
    n = 0
    batch_size = min(n_samples, cfg.get("eval_batch_size", 2000))
    with torch.no_grad():
        while n < n_samples:
            b = min(batch_size, n_samples - n)
            x0 = source.sample([b]).to(DEVICE)
            ts = get_timesteps(**ts_cfg).to(DEVICE)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            x1_list.append(x1)
            n += b

    samples = torch.cat(x1_list)[:n_samples]
    return samples, energy


def assign_modes(samples, centers):
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)
    return nearest.cpu().numpy()


def plot_dimension(d, runs):
    print(f"\n{'='*60}")
    print(f"Plotting RotGMM d={d}")
    print(f"{'='*60}")

    # Generate samples for all 3 methods
    print("  Loading baseline...")
    bl_samples, energy = load_and_generate(runs["baseline"], SEED, N_SAMPLES)
    print("  Loading RBF-KSD...")
    rbf_samples, _ = load_and_generate(runs["rbf_ksd"], SEED, N_SAMPLES)
    print("  Loading IMQ-KSD...")
    imq_samples, _ = load_and_generate(runs["imq_ksd"], SEED, N_SAMPLES)

    centers = energy.get_mode_centers().to(DEVICE)
    n_modes = centers.shape[0]
    ref_samples = energy.get_ref_samples().to(DEVICE)

    # Assign modes
    bl_assign = assign_modes(bl_samples, centers)
    rbf_assign = assign_modes(rbf_samples, centers)
    imq_assign = assign_modes(imq_samples, centers)
    ref_assign = assign_modes(ref_samples, centers)

    # ── Figure 1: Mode occupation bar chart (4 methods) ──
    print("  Plotting mode occupation bar chart...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))

    mode_indices = np.arange(n_modes)
    bar_width = 0.2

    bl_counts = np.array([(bl_assign == k).sum() for k in range(n_modes)])
    rbf_counts = np.array([(rbf_assign == k).sum() for k in range(n_modes)])
    imq_counts = np.array([(imq_assign == k).sum() for k in range(n_modes)])
    ref_counts = np.array([(ref_assign == k).sum() for k in range(n_modes)])

    bl_frac = bl_counts / max(bl_counts.sum(), 1)
    rbf_frac = rbf_counts / max(rbf_counts.sum(), 1)
    imq_frac = imq_counts / max(imq_counts.sum(), 1)
    ref_frac = ref_counts / max(ref_counts.sum(), 1)

    ax.bar(mode_indices - 1.5 * bar_width, ref_frac, bar_width,
           label="Reference", color="#AAAAAA", edgecolor="black", linewidth=0.5)
    ax.bar(mode_indices - 0.5 * bar_width, bl_frac, bar_width,
           label="Baseline ASBS", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(mode_indices + 0.5 * bar_width, rbf_frac, bar_width,
           label="KSD-ASBS (RBF)", color="#DD8452", edgecolor="black", linewidth=0.5)
    ax.bar(mode_indices + 1.5 * bar_width, imq_frac, bar_width,
           label="KSD-ASBS (IMQ)", color="#55A868", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Mode Index", fontsize=13)
    ax.set_ylabel("Fraction of Samples", fontsize=13)
    ax.set_title(f"RotGMM d={d}: Mode Occupation — Kernel Ablation (seed=0, N=2000)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(mode_indices)
    ax.set_xticklabels([f"Mode {k}" for k in range(n_modes)], fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    max_frac = max(bl_frac.max(), rbf_frac.max(), imq_frac.max(), ref_frac.max())
    ax.set_ylim(0, max_frac * 1.3)

    bl_covered = (bl_counts > 0).sum()
    rbf_covered = (rbf_counts > 0).sum()
    imq_covered = (imq_counts > 0).sum()
    y_text = 0.95
    for label, covered, color in [
        ("Baseline", bl_covered, "#4C72B0"),
        ("RBF-KSD", rbf_covered, "#DD8452"),
        ("IMQ-KSD", imq_covered, "#55A868"),
    ]:
        ax.text(0.02, y_text, f"{label}: {covered}/8 modes covered",
                transform=ax.transAxes, fontsize=10, va="top",
                color=color, fontweight="bold")
        y_text -= 0.065

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"rotgmm{d}_kernel_ablation_modes.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / f'rotgmm{d}_kernel_ablation_modes.png'}")
    plt.close()

    # ── Figure 2: PCA 2D scatter (4 panels) ──
    print("  Plotting PCA scatter...")

    pca = PCA(n_components=2)
    pca.fit(ref_samples.cpu().numpy())

    ref_2d = pca.transform(ref_samples.cpu().numpy())
    bl_2d = pca.transform(bl_samples.cpu().numpy())
    rbf_2d = pca.transform(rbf_samples.cpu().numpy())
    imq_2d = pca.transform(imq_samples.cpu().numpy())
    centers_2d = pca.transform(centers.cpu().numpy())

    cmap = plt.cm.tab10
    mode_colors = [cmap(k % 10) for k in range(n_modes)]

    def get_colors(assignments):
        return [mode_colors[a] for a in assignments]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    panels = [
        (axes[0], ref_2d, ref_assign, "Reference"),
        (axes[1], bl_2d, bl_assign, "Baseline ASBS"),
        (axes[2], rbf_2d, rbf_assign, "KSD-ASBS (RBF)"),
        (axes[3], imq_2d, imq_assign, "KSD-ASBS (IMQ)"),
    ]

    for ax, data_2d, assignments, title in panels:
        colors = get_colors(assignments)
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, s=8, alpha=0.5, rasterized=True)

        for k in range(n_modes):
            ax.scatter(centers_2d[k, 0], centers_2d[k, 1],
                       c=[mode_colors[k]], s=120, marker="*", edgecolors="black",
                       linewidths=0.8, zorder=5)
            ax.annotate(f"{k}", (centers_2d[k, 0], centers_2d[k, 1]),
                        fontsize=8, fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

        n_covered = len(set(a for a in assignments if a >= 0))
        ax.set_title(f"{title}\n({n_covered}/8 modes)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.tick_params(labelsize=8)

    all_2d = np.vstack([ref_2d, bl_2d, rbf_2d, imq_2d])
    margin = 1.5
    xlim = (all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin)
    ylim = (all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(f"RotGMM d={d}: PCA Projection — Kernel Ablation (seed=0, N=2000)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"rotgmm{d}_kernel_ablation_pca.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / f'rotgmm{d}_kernel_ablation_pca.png'}")
    plt.close()

    # Print summary
    print(f"\n  === d={d} Summary ===")
    print(f"  Baseline: {bl_covered}/8 modes, counts={bl_counts.tolist()}")
    print(f"  RBF-KSD:  {rbf_covered}/8 modes, counts={rbf_counts.tolist()}")
    print(f"  IMQ-KSD:  {imq_covered}/8 modes, counts={imq_counts.tolist()}")


def main():
    for d in DIMS:
        plot_dimension(d, RUNS[d])
    print("\n\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
