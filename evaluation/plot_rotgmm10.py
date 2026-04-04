"""
plot_rotgmm10.py — Generate RotGMM-10 visualization figures.

1. Mode occupation bar chart (baseline vs KSD-ASBS)
2. PCA 2D scatter plot colored by nearest mode

Uses seed=0, 2000 samples.
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

BASELINE_DIR = Path("/home/RESEARCH/Stein_ASBS/results/rotgmm10_asbs/seed_0")
KSD_DIR = Path("/home/RESEARCH/Stein_ASBS/results/rotgmm10_ksd_asbs/seed_0")


def load_and_generate(run_dir, seed, n_samples):
    """Load checkpoint and generate samples."""
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


# ── Generate samples ──
print("Loading baseline...")
bl_samples, energy = load_and_generate(BASELINE_DIR, SEED, N_SAMPLES)
print("Loading KSD-ASBS...")
ksd_samples, _ = load_and_generate(KSD_DIR, SEED, N_SAMPLES)

# ── Get mode info ──
centers = energy.get_mode_centers().to(DEVICE)  # (K, D)
n_modes = centers.shape[0]
ref_samples = energy.get_ref_samples().to(DEVICE)

# ── Assign samples to nearest mode ──
def assign_modes(samples, centers, threshold_factor=3.0, mode_std=0.5):
    dists = torch.cdist(samples, centers)  # (N, K)
    nearest = dists.argmin(dim=1)  # (N,)
    min_dists = dists.min(dim=1).values
    threshold = threshold_factor * mode_std
    # Mark samples too far from any mode as -1
    nearest[min_dists > threshold] = -1
    return nearest.cpu().numpy(), dists.cpu().numpy()

bl_assign, _ = assign_modes(bl_samples, centers)
ksd_assign, _ = assign_modes(ksd_samples, centers)
ref_assign, _ = assign_modes(ref_samples, centers)

# ── Figure 1: Mode occupation bar chart ──
print("Plotting mode occupation bar chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

mode_indices = np.arange(n_modes)
bar_width = 0.25

# Count fraction per mode
bl_counts = np.array([(bl_assign == k).sum() for k in range(n_modes)])
ksd_counts = np.array([(ksd_assign == k).sum() for k in range(n_modes)])
ref_counts = np.array([(ref_assign == k).sum() for k in range(n_modes)])

bl_frac = bl_counts / max(bl_counts.sum(), 1)
ksd_frac = ksd_counts / max(ksd_counts.sum(), 1)
ref_frac = ref_counts / max(ref_counts.sum(), 1)

bars_ref = ax.bar(mode_indices - bar_width, ref_frac, bar_width,
                  label="Reference", color="#AAAAAA", edgecolor="black", linewidth=0.5)
bars_bl = ax.bar(mode_indices, bl_frac, bar_width,
                 label="Baseline ASBS", color="#4C72B0", edgecolor="black", linewidth=0.5)
bars_ksd = ax.bar(mode_indices + bar_width, ksd_frac, bar_width,
                  label="KSD-ASBS (λ=1.0)", color="#DD8452", edgecolor="black", linewidth=0.5)

ax.set_xlabel("Mode Index", fontsize=13)
ax.set_ylabel("Fraction of Samples", fontsize=13)
ax.set_title("RotGMM d=10: Mode Occupation (seed=0, N=2000)", fontsize=14, fontweight="bold")
ax.set_xticks(mode_indices)
ax.set_xticklabels([f"Mode {k}" for k in range(n_modes)], fontsize=10)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, max(bl_frac.max(), ksd_frac.max(), ref_frac.max()) * 1.25)

# Annotate covered counts
bl_covered = (bl_counts > 0).sum()
ksd_covered = (ksd_counts > 0).sum()
ax.text(0.02, 0.95, f"Baseline: {bl_covered}/8 modes covered",
        transform=ax.transAxes, fontsize=11, va="top",
        color="#4C72B0", fontweight="bold")
ax.text(0.02, 0.88, f"KSD-ASBS: {ksd_covered}/8 modes covered",
        transform=ax.transAxes, fontsize=11, va="top",
        color="#DD8452", fontweight="bold")

plt.tight_layout()
fig.savefig(FIG_DIR / "rotgmm10_mode_occupation.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {FIG_DIR / 'rotgmm10_mode_occupation.png'}")
plt.close()


# ── Figure 2: PCA 2D scatter ──
print("Plotting PCA scatter...")

# Fit PCA on reference samples
all_data = torch.cat([ref_samples, bl_samples, ksd_samples], dim=0).cpu().numpy()
pca = PCA(n_components=2)
pca.fit(ref_samples.cpu().numpy())

ref_2d = pca.transform(ref_samples.cpu().numpy())
bl_2d = pca.transform(bl_samples.cpu().numpy())
ksd_2d = pca.transform(ksd_samples.cpu().numpy())
centers_2d = pca.transform(centers.cpu().numpy())

# Color map: mode assignment
cmap = plt.cm.tab10
mode_colors = [cmap(k % 10) for k in range(n_modes)]
unassigned_color = (0.8, 0.8, 0.8, 0.4)

def get_colors(assignments):
    return [mode_colors[a] if a >= 0 else unassigned_color for a in assignments]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, data_2d, assignments, title in [
    (axes[0], ref_2d, ref_assign, "Reference"),
    (axes[1], bl_2d, bl_assign, "Baseline ASBS"),
    (axes[2], ksd_2d, ksd_assign, "KSD-ASBS (λ=1.0)"),
]:
    colors = get_colors(assignments)
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, s=8, alpha=0.5, rasterized=True)

    # Plot mode centers
    for k in range(n_modes):
        ax.scatter(centers_2d[k, 0], centers_2d[k, 1],
                   c=[mode_colors[k]], s=120, marker="*", edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(f"{k}", (centers_2d[k, 0], centers_2d[k, 1]),
                    fontsize=8, fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    n_covered = len(set(assignments[assignments >= 0]))
    ax.set_title(f"{title}\n({n_covered}/8 modes covered)", fontsize=13, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.tick_params(labelsize=9)

# Use same axis limits across all three
all_2d = np.vstack([ref_2d, bl_2d, ksd_2d])
margin = 1.5
xlim = (all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin)
ylim = (all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin)
for ax in axes:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.suptitle("RotGMM d=10: PCA Projection (seed=0, N=2000)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "rotgmm10_pca_scatter.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {FIG_DIR / 'rotgmm10_pca_scatter.png'}")
plt.close()

print("\nDone!")
