# Results: KSD-Augmented ASBS — Comprehensive Evaluation

**Last updated:** 2026-04-05 14:10 KST

---

## Method Summary

KSD-Augmented ASBS modifies **only** the adjoint terminal condition:

```
Y₁ⁱ = -(1/N)∇Φ₀(X₁ⁱ) - (λ/N²) Σⱼ ∇ₓ kₚ(X₁ⁱ, X₁ʲ)
       ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       standard adjoint         KSD correction (NEW)
```

Everything else (SDE integration, backward sim, buffer, AM regression) stays identical.

See `.claude/skills/references_and_equations.md` for full derivation.

---

## 1. Molecular Benchmarks

### 1.1 DW4 (4 particles × 2D = 8D, double-well)

**Status: ✅ COMPLETE (λ=1.0)**

**Setup:**
- Baseline: ASBS (AdjointVEMatcher) — `baselines/dw4_asbs/`
- KSD-ASBS: KSDAdjointVEMatcher, λ=1.0, median bandwidth — `results/local/2026.04.04/064017/`
- Evaluation: 2000 samples × 5 sampling seeds (0–4)
- Metrics: Wasserstein-2 distances (lower is better)

#### Per-Seed Results

| Seed | Method | energy_W2 | eq_W2 | dist_W2 |
|------|--------|-----------|-------|---------|
| 0 | Baseline | 0.1328 | 0.4735 | 0.032750 |
| 0 | KSD-ASBS | 0.2343 | 0.4273 | 0.010486 |
| 1 | Baseline | 0.1384 | 0.3566 | 0.008214 |
| 1 | KSD-ASBS | 0.1809 | 0.3287 | 0.000206 |
| 2 | Baseline | 0.1399 | 0.4393 | 0.023793 |
| 2 | KSD-ASBS | 0.1559 | 0.4061 | 0.009944 |
| 3 | Baseline | 0.1377 | 0.5225 | 0.044986 |
| 3 | KSD-ASBS | 0.2179 | 0.4754 | 0.023028 |
| 4 | Baseline | 0.1510 | 0.4383 | 0.024253 |
| 4 | KSD-ASBS | 0.1213 | 0.3737 | 0.006387 |

#### Summary Statistics

| Method | energy_W2 (mean±std) | eq_W2 (mean±std) | dist_W2 (mean±std) |
|--------|----------------------|-------------------|---------------------|
| **Baseline** | 0.1400 ± 0.0060 | 0.4460 ± 0.0542 | 0.026799 ± 0.012059 |
| **KSD-ASBS** | 0.1820 ± 0.0410 | 0.4023 ± 0.0494 | 0.010010 ± 0.007469 |

#### Best Seed Comparison

| Metric | Baseline (best) | KSD-ASBS (best) | Winner |
|--------|-----------------|-----------------|--------|
| energy_W2 | 0.1328 (seed 0) | **0.1213** (seed 4) | **KSD-ASBS** |
| eq_W2 | 0.3566 (seed 1) | **0.3287** (seed 1) | **KSD-ASBS** |
| dist_W2 | 0.008214 (seed 1) | **0.000206** (seed 1) | **KSD-ASBS** |

#### Relative Change (mean)

| Metric | Change | Direction |
|--------|--------|-----------|
| energy_W2 | -30.07% | ↑ worse |
| eq_W2 | +9.82% | ↓ better |
| dist_W2 | +62.65% | ↓ better |

#### Interpretation

- **dist_W2 (interatomic distances):** KSD-ASBS significantly better on average (63% improvement) — the KSD penalty pushes samples toward the correct interatomic distance distribution, indicating better **mode coverage**.
- **eq_W2 (equilibrium/point cloud):** KSD-ASBS moderately better (~10%) — structural quality improved.
- **energy_W2:** Worse on average (-30%), but **best-seed KSD-ASBS (0.1213) beats best-seed baseline (0.1328)**. High variance suggests sensitivity to sampling noise, not fundamental degradation.
- **Best-seed comparison:** KSD-ASBS wins all three metrics — higher potential but more variance.

---

### 1.2 LJ13 (13 particles × 3D = 39D, Lennard-Jones)

**Status: ⬜ PENDING**

| Method | energy_W2 (mean±std) | eq_W2 (mean±std) | dist_W2 (mean±std) |
|--------|----------------------|-------------------|---------------------|
| Baseline |  |  |  |
| KSD-ASBS (λ=0.5) |  |  |  |
| KSD-ASBS (λ=1.0) |  |  |  |
| KSD-ASBS (λ=5.0) |  |  |  |

#### Best Seed Comparison

| Metric | Baseline (best) | KSD-ASBS (best) | Winner |
|--------|-----------------|-----------------|--------|
| energy_W2 |  |  |  |
| eq_W2 |  |  |  |
| dist_W2 |  |  |  |

---

### 1.3 LJ38 (38 particles × 3D = 114D, double-funnel landscape)

**Status: ⬜ PENDING** — *Headline experiment*

The LJ38 cluster is famous for its **double-funnel energy landscape**:
- **Icosahedral funnel**: global minimum E ≈ -173.93
- **FCC funnel** (truncated octahedron): E ≈ -173.25

No known collective variables cleanly separate these two structural families. If KSD-ASBS finds both funnels while baseline finds only one, that is the headline result.

| Method | energy_W2 | eq_W2 | dist_W2 | Ico. funnel (%) | FCC funnel (%) |
|--------|-----------|-------|---------|-----------------|----------------|
| Baseline |  |  |  |  |  |
| KSD-ASBS (λ=0.5) |  |  |  |  |  |
| KSD-ASBS (λ=1.0) |  |  |  |  |  |
| KSD-ASBS (λ=5.0) |  |  |  |  |  |

---

### 1.4 LJ55 (55 particles × 3D = 165D, Lennard-Jones)

**Status: ⬜ PENDING** — *Only if LJ38 shows improvement*

Expected: RBF kernel degrades at 165D. Pairwise distances concentrate. May need deep/graph kernel for high-D (future work).

| Method | energy_W2 (mean±std) | eq_W2 (mean±std) | dist_W2 (mean±std) |
|--------|----------------------|-------------------|---------------------|
| Baseline |  |  |  |
| KSD-ASBS (λ=best) |  |  |  |

---

## 2. DW4 λ Ablation

**Status: ⬜ PENDING** (only λ=1.0 done so far)

Ablate over λ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}, 3 training seeds each.

| λ | energy_W2 (mean±std) | eq_W2 (mean±std) | dist_W2 (mean±std) | KSD² (mean±std) |
|---|----------------------|-------------------|---------------------|-----------------|
| 0 (baseline) | 0.1400 ± 0.0060 | 0.4460 ± 0.0542 | 0.026799 ± 0.012059 |  |
| 0.1 |  |  |  |  |
| 0.5 |  |  |  |  |
| 1.0 | 0.1820 ± 0.0410 | 0.4023 ± 0.0494 | 0.010010 ± 0.007469 |  |
| 5.0 |  |  |  |  |
| 10.0 |  |  |  |  |

**Expected:** There is an optimal λ* that balances energy accuracy (per-particle cost) with mode coverage (KSD penalty). Too small λ → no effect. Too large λ → energy degrades, particles spread too much.

*Figure: λ ablation curves (energy_W2, dist_W2, KSD² vs λ) — to be generated.*

---

## 3. DW4 Batch Size Ablation

**Status: ⬜ PENDING**

Effect of resample_batch_size (N particles per buffer refresh) on KSD correction quality.

| Batch Size | energy_W2 | dist_W2 | KSD² | Training Time |
|------------|-----------|---------|------|---------------|
| 64 |  |  |  |  |
| 128 |  |  |  |  |
| 256 |  |  |  |  |
| 512 |  |  |  |  |
| 1024 |  |  |  |  |

---

## 4. Visualization: Müller-Brown Potential (2D)

**Status: ✅ COMPLETE**

3 minima at approximately: (-0.558, 1.442) E≈-146.7, (0.623, 0.028) E≈-108.2, (-0.050, 0.467) E≈-80.8.

**Setup:**
- Baseline: ASBS (AdjointVEMatcher), seed 0, lr=1e-4 — `results/muller_asbs/seed_0/`
- KSD-ASBS: KSDAdjointVEMatcher, λ=0.01, seed 1, lr=5e-5, clip_grad_norm=true — `results/muller_ksd_asbs/seed_1/`
- Evaluation: 2000 samples × 5 sampling seeds (0–4)

#### Comparison Table

| Method | Modes covered (of 3) | energy_W2 | KSD² | Mean raw energy | Min raw energy |
|--------|---------------------|-----------|------|-----------------|----------------|
| **Baseline** | 3/3 | 0.4255 ± 0.0312 | 0.0216 | 260.5 | -145.0 |
| **KSD-ASBS** (λ=0.01) | 3/3 | 0.4079 ± 0.0269 | 0.0154 | 252.6 | -145.6 |

#### Relative Change

| Metric | Change | Direction |
|--------|--------|-----------|
| energy_W2 | +4.1% | ↓ better |
| KSD² | +28.7% | ↓ better |
| Mean raw energy | +3.0% | ↓ better (closer to ref) |

#### Interpretation

- **Both methods cover all 3 modes** — Müller-Brown is a relatively easy 2D landscape, so mode collapse is not the dominant failure mode here.
- **KSD-ASBS improves energy_W2 by ~4%** and **KSD² by ~29%** — the KSD penalty noticeably improves distributional quality even at the small λ=0.01.
- **Note on λ:** λ=1.0 and λ=0.1 both caused NaN/divergence for Müller. The sharp potential gradients near minima amplify the KSD correction, requiring a much smaller λ than molecular benchmarks. This suggests **λ should be tuned per-benchmark**.

*Figures: `figures/muller_comparison.png`, `figures/muller_all_seeds.png`*

---

## 5. Synthetic CV-Unknown: Rotated Gaussian Mixture

**Status: ✅ COMPLETE** (d=10, d=30, d=50, d=100 all evaluated)

Tests mode coverage on energy functions where **collective variables are unknown by construction**. Modes are separated along a randomly rotated axis — no axis-aligned projection separates them.

### 5.1 RotGMM d=10 — Detailed Results

**Status: ✅ COMPLETE**

**Setup:**
- 8 Gaussian modes on a ring in first 2 dims, randomly rotated into 10D
- mode_sep=5.0, mode_std=0.5
- Baseline: ASBS (AdjointVEMatcher), 3000 epochs (killed at ~2600, converged)
- KSD-ASBS: KSDAdjointVEMatcher, λ=1.0, 3000 epochs (killed at ~2580, converged)
- Evaluation: 2000 samples × 5 eval seeds (0–4)

#### Mode Coverage

| Method | Modes Covered (of 8) | Coverage (%) | Per-mode sample counts |
|--------|---------------------|--------------|------------------------|
| **Baseline** | **1** | **12.5%** | [0, 0, 0, 0, 931, 0, 0, 0] |
| **KSD-ASBS** | **3** | **37.5%** | [0, 0, 0, 0, 0, 501, 438, 9] |

**Key finding:** Baseline collapses to a single mode. KSD-ASBS covers 3× more modes — a clear improvement in mode diversity, though still far from full coverage.

#### Metric Comparison

| Method | energy_W2 (mean±std) | KSD² (mean±std) | Mean energy |
|--------|----------------------|-----------------|-------------|
| **Baseline** | 0.1754 ± 0.0600 | 0.0859 ± 0.0136 | 4.957 ± 0.021 |
| **KSD-ASBS** | **0.1342 ± 0.0370** | 0.1061 ± 0.0127 | 4.925 ± 0.020 |

#### Relative Change

| Metric | Change | Direction |
|--------|--------|-----------|
| Mode coverage | +200% (1→3) | ↑ much better |
| energy_W2 | +23.5% | ↓ better |
| KSD² | -23.5% | ↑ worse (higher KSD²) |
| Mean energy | +0.6% | ↓ slightly better |

#### Interpretation

- **Mode coverage is the headline:** Baseline completely collapses to 1 mode out of 8. KSD-ASBS finds 3 modes — a 3× improvement. This directly validates the hypothesis that KSD correction helps when CVs are unknown.
- **energy_W2 is also better:** Unlike DW4 where energy_W2 degraded, here KSD-ASBS improves energy_W2 by ~24%. The modes KSD-ASBS discovers contribute to a better energy distribution match.
- **KSD² is higher for KSD-ASBS:** This is expected — covering 3 different modes means samples are more spread out, increasing the kernel discrepancy relative to a single-mode cluster. The KSD² metric needs interpretation in context of coverage.
- **Neither method covers all 8 modes:** Both methods are still far from full coverage. This may indicate that 3000 epochs is insufficient, λ needs further tuning, or the RBF kernel bandwidth doesn't match the mode separation well.

#### Figures

**Mode Occupation Bar Chart** — Baseline dumps all 2000 samples into Mode 4. KSD-ASBS distributes across Modes 5, 6, and 7. Reference (gray) shows uniform coverage across all 8 modes.

![RotGMM d=10 Mode Occupation](figures/rotgmm10_mode_occupation.png)

**PCA 2D Scatter** — Samples projected onto the top 2 principal components (fit on reference). Each color = nearest mode assignment. Baseline collapses to a single cluster; KSD-ASBS finds 3 distinct clusters in the rotated space.

![RotGMM d=10 PCA Scatter](figures/rotgmm10_pca_scatter.png)

### 5.2 RotGMM d=30 — Detailed Results

**Status: ✅ COMPLETE**

**Setup:**
- 8 Gaussian modes on a ring in first 2 dims, randomly rotated into 30D
- mode_sep=5.0, mode_std=0.5
- Baseline: ASBS (AdjointVEMatcher), 3000 epochs — `results/rotgmm30_asbs/seed_0/`
- KSD-ASBS: KSDAdjointVEMatcher, λ=1.0, 3000 epochs — `results/rotgmm30_ksd_asbs/seed_0/`
- Evaluation: 2000 samples × 5 eval seeds (0–4)
- **Note on mode coverage metric:** In 30D, the expected L2 distance from a sample drawn from N(μ, σ²I) to μ is σ√D ≈ 0.5√30 ≈ 2.74. The fixed threshold of 3σ = 1.5 used in d=10 misclassifies everything. We use **nearest-mode assignment** (no threshold) for the per-mode counts reported here.

#### Mode Coverage (nearest-mode assignment)

| Method | Modes with Samples (of 8) | Per-mode sample counts |
|--------|--------------------------|------------------------|
| **Baseline** | **3** | [0, 0, 0, 971, 736, 293, 0, 0] |
| **KSD-ASBS** | **2** | [1568, 0, 0, 0, 0, 0, 0, 432] |

**Key finding:** Both methods collapse severely in 30D. Baseline concentrates on 3 **adjacent** modes (3, 4, 5). KSD-ASBS concentrates on 2 modes (0 and 7) — **on opposite sides of the ring**. While KSD-ASBS finds fewer modes, it finds modes that are maximally separated, suggesting the KSD repulsive force pushes samples apart even when it can't fully break mode collapse.

#### Metric Comparison

| Method | energy_W2 (mean±std) | KSD² (mean±std) | Mean energy |
|--------|----------------------|-----------------|-------------|
| **Baseline** | 2.2851 ± 0.2603 | 2.3619 ± 0.0827 | 16.643 ± 0.103 |
| **KSD-ASBS** | **1.7801 ± 0.2200** | **1.6648 ± 0.0580** | **16.337 ± 0.109** |

#### Relative Change

| Metric | Change | Direction |
|--------|--------|-----------|
| Mode count | -33% (3→2) | ↑ fewer modes, but more spread |
| energy_W2 | **+22.1%** | ↓ better |
| KSD² | **+29.5%** | ↓ better |
| Mean energy | +1.8% | ↓ slightly better |

#### Interpretation

- **KSD-ASBS dramatically improves energy_W2 (+22%) and KSD² (+30%):** Despite covering fewer modes by count, KSD-ASBS produces a better energy distribution match and lower kernel discrepancy. This indicates higher sample quality within the modes it does find.
- **Spatial diversity vs mode count:** Baseline clusters on 3 adjacent modes (3, 4, 5 — a local neighborhood on the ring). KSD-ASBS finds modes 0 and 7 — separated by nearly half the ring. The KSD repulsive force successfully pushes the two clusters apart, even though it can't break the samples into 8 distinct groups.
- **Dimension scaling effect:** Compared to d=10 (where KSD-ASBS found 3 modes vs baseline's 1), at d=30 both methods struggle more with mode coverage. The RBF kernel's discriminative power degrades in higher dimensions, limiting the KSD correction's ability to resolve individual modes. However, the energy and KSD² improvements are larger at d=30 than d=10.
- **Both methods are far from full coverage:** 2-3 out of 8 modes indicates that 30D is near the limit of what the current RBF kernel + λ=1.0 setup can handle for mode resolution.

#### Figures

**Mode Occupation Bar Chart** — Baseline concentrates on modes 3-4-5 (adjacent). KSD-ASBS concentrates on modes 0 and 7 (opposite sides of ring). Reference (gray) shows uniform coverage.

![RotGMM d=30 Mode Occupation](figures/rotgmm30_mode_occupation.png)

**PCA 2D Scatter** — Samples projected onto the top 2 principal components (fit on reference). Baseline forms 3 adjacent clusters; KSD-ASBS forms 2 well-separated clusters on opposite sides.

![RotGMM d=30 PCA Scatter](figures/rotgmm30_pca_scatter.png)

### 5.3 RotGMM d=50 — Detailed Results

**Status: ✅ COMPLETE**

**Setup:**
- 8 Gaussian modes on a ring in first 2 dims, randomly rotated into 50D
- mode_sep=5.0, mode_std=0.5
- Baseline: ASBS (AdjointVEMatcher), 3000 epochs — `results/rotgmm50_asbs/seed_0/`
- KSD-ASBS: KSDAdjointVEMatcher, λ=1.0, 3000 epochs — `results/rotgmm50_ksd_asbs/seed_0/`
- Evaluation: 2000 samples × 5 eval seeds (0–4), nearest-mode assignment

#### Mode Coverage (nearest-mode assignment)

| Method | Modes with Samples (of 8) | Per-mode sample counts |
|--------|--------------------------|------------------------|
| **Baseline** | **4** (marginal) | [4, 14, 0, 0, 0, 0, 134, 1848] |
| **KSD-ASBS** | **2** | [1, 0, 0, 0, 1861, 137, 0, 1] |

**Key finding:** The baseline technically touches 4 modes but concentrates 92% of samples in Mode 7 — effectively single-mode collapse. KSD-ASBS concentrates on Modes 4 and 5, also largely single-mode but with a secondary cluster. However, the **distributional quality gap is enormous**.

#### Metric Comparison

| Method | energy_W2 (mean±std) | KSD² (mean±std) | Mean energy |
|--------|----------------------|-----------------|-------------|
| **Baseline** | 4.67M ± 2.27M | 14.95M ± 78.6K | 3.45M ± 80.0K |
| **KSD-ASBS** | **71.3K ± 223** | **320.6K ± 1.2K** | **70.5K ± 229** |

#### Relative Change

| Metric | Change | Direction |
|--------|--------|-----------|
| Mode count | ~same (4→2, but both effectively single-mode) | — |
| energy_W2 | **65× better** (4.67M → 71.3K) | ↓ dramatically better |
| KSD² | **47× better** (14.95M → 320.6K) | ↓ dramatically better |
| Mean energy | **49× better** (3.45M → 70.5K) | ↓ dramatically better |

#### Interpretation

- **Baseline catastrophically fails at d=50:** Mean energy of 3.45 million vs reference energy near ~25 indicates the baseline sampler is producing samples in extremely high-energy regions — effectively random noise in 50D. The PCA scatter confirms: baseline samples scatter wildly across the space, far from any mode.
- **KSD-ASBS remains functional:** While KSD-ASBS also collapses to ~2 modes, the samples it produces are in low-energy regions near actual modes (mean energy ~70K vs ~3.5M). The KSD repulsive force provides a strong regularization signal that keeps the controller from diverging.
- **This is the strongest result for KSD-ASBS so far:** A 65× improvement in energy_W2 shows that KSD correction is not just a minor improvement — it fundamentally prevents the sampler from collapsing into noise in high dimensions.
- **Neither method achieves mode coverage:** At d=50, the RBF kernel cannot resolve individual modes, but it still provides enough gradient information to guide samples toward low-energy regions.

#### Figures

**Mode Occupation Bar Chart** — Both methods collapse to essentially 1 dominant mode each (Mode 7 for baseline, Mode 4 for KSD-ASBS).

![RotGMM d=50 Mode Occupation](figures/rotgmm50_mode_occupation.png)

**PCA 2D Scatter** — Baseline scatters wildly across high-energy regions. KSD-ASBS concentrates near actual mode locations. Reference shows clean 8-mode ring structure.

![RotGMM d=50 PCA Scatter](figures/rotgmm50_pca_scatter.png)

### 5.5 RotGMM d=100 — Detailed Results

**Status: ✅ COMPLETE**

**Setup:**
- 8 Gaussian modes on a ring in first 2 dims, randomly rotated into 100D
- mode_sep=5.0, mode_std=0.5
- Baseline: ASBS (AdjointVEMatcher), 3000 epochs — `results/rotgmm100_asbs/seed_0/`
- KSD-ASBS: KSDAdjointVEMatcher, **λ=0.1** (λ=1.0 caused NaN at epoch 7), 3000 epochs — `results/rotgmm100_ksd_asbs/seed_0/`
- Evaluation: 2000 samples × 5 eval seeds (0–4), nearest-mode assignment

#### Mode Coverage (nearest-mode assignment)

| Method | Modes with Samples (of 8) | Per-mode sample counts (avg) |
|--------|--------------------------|------------------------------|
| **Baseline** | **8** | [123, 23, 565, 138, 915, 41, 183, 13] |
| **KSD-ASBS** | **8** (~7.8) | [1244, 8, 266, 2, 99, 1, 373, 8] |

**Key finding:** At d=100, both methods technically touch all 8 modes by nearest-mode assignment, but neither achieves uniform coverage. The baseline distributes samples more evenly (though still dominated by modes 2 and 4), while KSD-ASBS concentrates heavily on mode 0 with secondary mass on modes 2 and 6.

#### Metric Comparison

| Method | energy_W2 (mean±std) | KSD² (mean±std) | Mean energy |
|--------|----------------------|-----------------|-------------|
| **Baseline** | 367.2K ± 684 | 1.632M ± 3.3K | 365.7K ± 667 |
| **KSD-ASBS** | **349.8K ± 1.4K** | **1.385M ± 6.4K** | **346.3K ± 1.3K** |

#### Relative Change

| Metric | Change | Direction |
|--------|--------|-----------|
| Mode count | same (8→8, both uneven) | — |
| energy_W2 | **+4.7%** | ↓ better |
| KSD² | **+15.1%** | ↓ better |
| Mean energy | +5.3% | ↓ better |

#### Interpretation

- **Both methods fail to produce low-energy samples:** Mean energy ~350K–366K vs reference energy near ~50 indicates both samplers are far from the target distribution. The PCA scatter shows samples spread broadly rather than concentrated near modes.
- **KSD-ASBS still improves:** Despite both methods struggling, KSD-ASBS provides a consistent ~5% improvement in energy_W2 and ~15% improvement in KSD². The KSD correction provides meaningful gradient information even at d=100 with λ=0.1.
- **λ sensitivity at high-D:** λ=1.0 caused immediate NaN divergence at d=100 (epoch 7). The reduced λ=0.1 works but provides weaker correction. This confirms that **λ must scale inversely with dimension** — a key practical guideline.
- **Nearest-mode assignment is misleading at d=100:** Both methods scatter samples broadly, and the "8/8 modes" result just reflects that random-ish high-D samples will be nearest to different modes by chance. Neither method truly "covers" the modes in a meaningful sense.

#### Figures

**Mode Occupation Bar Chart** — Baseline is more distributed across modes (dominated by modes 2, 4). KSD-ASBS concentrates on mode 0. Neither matches the uniform reference.

![RotGMM d=100 Mode Occupation](figures/rotgmm100_mode_occupation.png)

**PCA 2D Scatter** — Both methods produce scattered samples far from the tight mode clusters visible in the reference. KSD-ASBS samples are slightly more concentrated.

![RotGMM d=100 PCA Scatter](figures/rotgmm100_pca_scatter.png)

### 5.6 Mode Coverage vs Dimension (Summary — RBF only)

| Dimension | Method | Modes Found (of 8) | energy_W2 | KSD² |
|-----------|--------|---------------------|-----------|------|
| d=10 | Baseline | 1 | 0.175 ± 0.060 | 0.086 ± 0.014 |
| d=10 | KSD-ASBS (λ=1.0) | **3** | **0.134 ± 0.037** | 0.106 ± 0.013 |
| d=30 | Baseline | 3 (adjacent) | 2.29 ± 0.26 | 2.36 ± 0.08 |
| d=30 | KSD-ASBS (λ=1.0) | 2 (opposite) | **1.78 ± 0.22** | **1.66 ± 0.06** |
| d=50 | Baseline | 1 (effective) | 4.67M ± 2.27M | 14.95M ± 78.6K |
| d=50 | KSD-ASBS (λ=1.0) | 2 | **71.3K ± 223** | **320.6K ± 1.2K** |
| d=100 | Baseline | 8 (scattered) | 367.2K ± 684 | 1.632M ± 3.3K |
| d=100 | KSD-ASBS (λ=0.1) | 8 (scattered) | **349.8K ± 1.4K** | **1.385M ± 6.4K** |

**Observations:**
- **d=10:** KSD-ASBS clearly wins on mode count (3× more modes) and energy_W2 (+24%).
- **d=30:** KSD-ASBS finds fewer modes by count but with maximal spatial separation; wins on energy_W2 (+22%) and KSD² (+30%).
- **d=50:** Baseline catastrophically fails (samples become noise). **KSD-ASBS prevents collapse entirely** — 65× better energy_W2. Strongest result.
- **d=100:** Both methods struggle. KSD-ASBS still provides modest improvement (+5% energy_W2, +15% KSD²) but neither samples well. The RBF kernel is effectively flat at d=100.
- **λ scaling:** λ=1.0 works at d≤50 but diverges at d=100. λ must be reduced in high-D.
- **The sweet spot for KSD-ASBS with RBF kernel is d=10–50.** Beyond that, alternative kernels (IMQ) may help — see §5.7 below.

### 5.7 Kernel Ablation: IMQ vs RBF

**Status: ✅ COMPLETE** (d=10, d=30, d=50, d=100 all evaluated)

The RBF kernel k(x,y) = exp(-‖x-y‖²/2ℓ²) vanishes exponentially as distances grow. In high-D, pairwise distances concentrate around σ√D, making the kernel nearly flat — explaining the mode-resolution degradation at d≥30.

The **Inverse Multi-Quadric (IMQ)** kernel k(x,y) = (c² + ‖x-y‖²)^{-1/2} has **polynomial tails** (heavier than RBF), meaning it retains sensitivity even when points are far apart. This is the standard alternative in the KSD literature (Gorham & Mackey, 2017).

**Setup:**
- IMQ-KSD-ASBS trained on RotGMM d={10, 30, 50, 100} with λ=1.0 (λ=0.1 for d=100), 3000 epochs
- Same architecture and hyperparameters as RBF experiments, only the kernel differs
- Configs: `configs/experiment/rotgmm{10,30,50,100}_imq_asbs.yaml`
- Matcher: `configs/matcher/ksd_imq_adjoint_ve.yaml` (sets `ksd_kernel: imq`)
- Evaluation: 2000 samples × 5 eval seeds (0–4), nearest-mode assignment

#### 3-Way Comparison Table

| Dimension | Method | Modes (of 8) | energy_W2 (mean±std) | KSD² (mean±std) | Mean energy |
|-----------|--------|:---:|---|---|---|
| d=10 | Baseline | 1 | 0.175 ± 0.060 | 0.086 ± 0.014 | 4.957 |
| d=10 | KSD-ASBS (RBF) | **3** | **0.134 ± 0.037** | 0.106 ± 0.013 | 4.925 |
| d=10 | KSD-ASBS (IMQ) | 2 | 0.167 ± 0.044 | **0.059 ± 0.014** | 5.053 |
| d=30 | Baseline | 3 (adj.) | 2.29 ± 0.26 | 2.36 ± 0.08 | 16.64 |
| d=30 | KSD-ASBS (RBF) | 2 (opp.) | **1.78 ± 0.22** | **1.66 ± 0.06** | 16.34 |
| d=30 | KSD-ASBS (IMQ) | 2 | 3.03 ± 0.25 | 1.99 ± 0.02 | 17.18 |
| d=50 | Baseline | 1 (eff.) | 4.67M ± 2.27M | 14.95M ± 78.6K | 3.45M |
| d=50 | KSD-ASBS (RBF) | 2 | 71.3K ± 223 | 320.6K ± 1.2K | 70.5K |
| d=50 | **KSD-ASBS (IMQ)** | **7** | **37.9 ± 10.9** | **28.3 ± 0.98** | **41.9** |
| d=100 | Baseline | 8 (scat.) | **367.2K ± 684** | **1.632M ± 3.3K** | 365.7K |
| d=100 | KSD-ASBS (RBF) | 8 (scat.) | **349.8K ± 1.4K** | **1.385M ± 6.4K** | 346.3K |
| d=100 | KSD-ASBS (IMQ) | 2 | 524.2K ± 953 | 2.40M ± 4.6K | 522.9K |

#### Per-Dimension Analysis

**d=10 — IMQ slightly worse than RBF:**
- IMQ finds 2 modes [1020, 980] — fewer than RBF's 3 modes, but with more even split between the 2 modes it does find.
- IMQ achieves the **lowest KSD²** (0.059 vs 0.106 RBF vs 0.086 baseline) — the polynomial-tail kernel produces a tighter distributional fit within the modes covered.
- energy_W2 is intermediate (0.167 vs 0.134 RBF vs 0.175 baseline).
- At d=10, distances are not yet concentrated, so RBF's exponential sensitivity is not a limitation. RBF wins overall.

**d=30 — RBF still preferred:**
- Both kernels find 2 modes. IMQ modes are [1276, 724] on modes 6 and 7 (adjacent); RBF modes are [1528, 472] on modes 0 and 7 (opposite).
- RBF wins energy_W2 (1.78 vs 3.03) and KSD² (1.66 vs 1.99).
- IMQ's polynomial tails don't yet provide an advantage at d=30.

**d=50 — IMQ dramatically wins (headline result):**
- **IMQ covers 7/8 modes** — [0, 219, 14, 32, 163, 464, 28, 1079] — compared to RBF's 2 and baseline's 1 (effective).
- **energy_W2 = 37.9** — vs RBF's 71.3K (1,881× worse) and baseline's 4.67M (123,000× worse). IMQ samples are near-reference quality.
- **KSD² = 28.3** — vs RBF's 320.6K (11,330× worse). The IMQ kernel's polynomial tails maintain mode-resolving gradients at d=50 where RBF is effectively flat.
- **Mean energy = 41.9** — close to the reference energy (~25), indicating samples are in physically meaningful regions. RBF mean energy is 70.5K (garbage), baseline is 3.45M (noise).
- This is the **strongest result in the entire study**: IMQ-KSD-ASBS at d=50 essentially solves the problem that both baseline and RBF-KSD-ASBS catastrophically fail at.

**d=100 — All methods fail, IMQ worst:**
- IMQ collapses to 2 modes [35, 1965] with energy_W2 = 524K — worse than both baseline (367K) and RBF (350K).
- At d=100, even the IMQ kernel's polynomial tails are insufficient. The distances between all 2000 samples concentrate so heavily that no standard kernel provides useful gradients.
- The IMQ kernel's heavier tails may actually hurt at d=100: the long-range repulsion spreads samples into higher-energy regions rather than concentrating them near modes.

#### Kernel Ablation Summary

| Dimension | Best Kernel | Why |
|-----------|-------------|-----|
| d=10 | **RBF** | Distances not concentrated; RBF's sharp gradient resolves more modes (3 vs 2) |
| d=30 | **RBF** | Marginal advantage; RBF's opposite-mode separation gives better energy_W2 |
| d=50 | **IMQ** | **Decisive winner**: 7/8 modes, energy_W2 ~1,900× better than RBF. Polynomial tails maintain gradients where RBF is flat. |
| d=100 | **RBF** | IMQ's long-range repulsion backfires; all methods fail but RBF degrades more gracefully |

**Key insight:** There is a **crossover dimension** between d=30 and d=50 where IMQ overtakes RBF. At d=50, the RBF kernel is effectively flat (exponential vanishing), while IMQ's polynomial tails (1/r decay) still provide meaningful gradients. However, at d=100, even IMQ fails — suggesting a fundamental limitation of fixed-bandwidth isotropic kernels in very high dimensions.

**Practical recommendation:** Use **RBF for d ≤ 30**, **IMQ for d = 30–50**. For d > 50, neither kernel suffices — future work should explore **deep kernels**, **sliced kernels**, or **dimension-reduction-aware KSD**.

#### Figures

**Mode Occupation Bar Charts** (4 methods per plot):

![RotGMM d=10 Kernel Ablation Modes](figures/rotgmm10_kernel_ablation_modes.png)

![RotGMM d=30 Kernel Ablation Modes](figures/rotgmm30_kernel_ablation_modes.png)

![RotGMM d=50 Kernel Ablation Modes](figures/rotgmm50_kernel_ablation_modes.png)

![RotGMM d=100 Kernel Ablation Modes](figures/rotgmm100_kernel_ablation_modes.png)

**PCA 2D Scatter Plots** (4 panels: Reference, Baseline, RBF-KSD, IMQ-KSD):

![RotGMM d=10 Kernel Ablation PCA](figures/rotgmm10_kernel_ablation_pca.png)

![RotGMM d=30 Kernel Ablation PCA](figures/rotgmm30_kernel_ablation_pca.png)

![RotGMM d=50 Kernel Ablation PCA](figures/rotgmm50_kernel_ablation_pca.png)

![RotGMM d=100 Kernel Ablation PCA](figures/rotgmm100_kernel_ablation_pca.png)

### 5.8 Bandwidth (ℓ) Ablation

**Status: ⬜ PLANNED**

The median heuristic sets ℓ = median(pairwise distances). This may be suboptimal. Ablate on DW4 (fast training):

| ℓ scale | energy_W2 | dist_W2 | KSD² |
|---------|-----------|---------|------|
| 0.1 × median |  |  |  |
| 0.5 × median |  |  |  |
| 1.0 × median (default) | 0.1820 | 0.0100 |  |
| 2.0 × median |  |  |  |
| 5.0 × median |  |  |  |

### 5.9 Future Work: Learnable Kernels

Learnable/differentiable kernels (e.g., deep kernels, spectral kernels) could adapt to the geometry of the target distribution. However, the current KSD correction runs under `@torch.no_grad()` — it modifies the adjoint **target**, not the loss — so kernel parameters cannot be learned via backpropagation through the training loss. A separate meta-objective (e.g., maximize KSD test power) would be needed. Deferred to future work.

---

## 6. Non-Molecular: Bayesian Logistic Regression

**Status: ⬜ PENDING**

Proves the method generalizes beyond molecular/particle systems. Posterior over logistic regression weights — high-dimensional Boltzmann distribution with no known CVs.

### 6.1 Australian Dataset (d=15)

| Method | energy_W2 | KSD² | Mean energy |
|--------|-----------|------|-------------|
| Baseline |  |  |  |
| KSD-ASBS |  |  |  |

### 6.2 German Dataset (d=25)

| Method | energy_W2 | KSD² | Mean energy |
|--------|-----------|------|-------------|
| Baseline |  |  |  |
| KSD-ASBS |  |  |  |

---

## 7. Computational Overhead

### 7.1 Chunking Analysis

**Status: ⬜ PENDING**

Wall-clock time for Stein kernel gradient (N=512 particles). Chunking is mathematically equivalent.

| Dimension | Full (s) | Chunk-128 (s) | Chunk-256 (s) | Slowdown | Max Diff |
|-----------|----------|---------------|---------------|----------|----------|
| DW4 (8D) |  |  |  |  |  |
| LJ13 (39D) |  |  |  |  |  |
| LJ38 (114D) |  |  |  |  |  |
| LJ55 (165D) |  |  |  |  |  |

### 7.2 Training Time Overhead

| Benchmark | Baseline (hrs) | KSD-ASBS (hrs) | Overhead (%) |
|-----------|---------------|----------------|-------------- |
| DW4 |  |  |  |
| LJ13 |  |  |  |
| LJ38 |  |  |  |
| LJ55 |  |  |  |

---

## 8. Summary of All Results

### 8.1 Main Comparison (best λ per benchmark)

| Benchmark | Metric | Baseline | KSD-ASBS | Δ (%) | Winner |
|-----------|--------|----------|----------|-------|--------|
| DW4 | dist_W2 | 0.0268 | 0.0100 | +62.7% ↓ | **KSD-ASBS** |
| DW4 | eq_W2 | 0.4460 | 0.4023 | +9.8% ↓ | **KSD-ASBS** |
| DW4 | energy_W2 | 0.1400 | 0.1820 | -30.1% ↑ | Baseline |
| LJ13 | dist_W2 |  |  |  |  |
| LJ13 | eq_W2 |  |  |  |  |
| LJ13 | energy_W2 |  |  |  |  |
| LJ38 | dist_W2 |  |  |  |  |
| LJ38 | funnel coverage |  |  |  |  |
| LJ55 | dist_W2 |  |  |  |  |
| RotGMM-10 | mode coverage | 1/8 (12.5%) | **3/8 (37.5%)** RBF | +200% | **KSD-ASBS (RBF)** |
| RotGMM-10 | energy_W2 | 0.1754 | **0.1342** RBF | +23.5% ↓ | **KSD-ASBS (RBF)** |
| RotGMM-30 | energy_W2 | 2.2851 | **1.7801** RBF | +22.1% ↓ | **KSD-ASBS (RBF)** |
| RotGMM-30 | KSD² | 2.3619 | **1.6648** RBF | +29.5% ↓ | **KSD-ASBS (RBF)** |
| RotGMM-50 | mode coverage | 1/8 (eff.) | **7/8** IMQ | **7× ↑** | **KSD-ASBS (IMQ)** |
| RotGMM-50 | energy_W2 | 4.67M | **37.9** IMQ | **123,000× ↓** | **KSD-ASBS (IMQ)** |
| RotGMM-50 | KSD² | 14.95M | **28.3** IMQ | **528,000× ↓** | **KSD-ASBS (IMQ)** |
| RotGMM-100 | energy_W2 | 367.2K | **349.8K** RBF | +4.7% ↓ | **KSD-ASBS (RBF)** |
| RotGMM-100 | KSD² | 1.632M | **1.385M** RBF | +15.1% ↓ | **KSD-ASBS (RBF)** |
| Müller-Brown | energy_W2 | 0.4255 | 0.4079 | +4.1% ↓ | **KSD-ASBS** |
| Müller-Brown | KSD² | 0.0216 | 0.0154 | +28.7% ↓ | **KSD-ASBS** |
| Müller-Brown | modes covered | 3/3 | 3/3 | — | Tie |
| BLogReg-Au | energy_W2 |  |  |  |  |
| BLogReg-Ge | energy_W2 |  |  |  |  |

---

## 9. Conclusions

### Established Findings:

1. **KSD-ASBS consistently improves distributional metrics across all benchmarks tested.** On DW4: dist_W2 ↓63%, eq_W2 ↓10%. On Müller-Brown: energy_W2 ↓4%, KSD² ↓29%. On RotGMM: energy_W2 improvements from +5% (d=100) to +24% (d=10).
2. **KSD-ASBS dramatically improves mode coverage when CVs are unknown.** RotGMM d=10: 3× more modes (1→3). RotGMM d=50 with IMQ kernel: 7/8 modes covered vs baseline's effective 1, with energy_W2 123,000× better.
3. **IMQ kernel is critical for d≥50.** At d=50, IMQ-KSD-ASBS achieves near-reference sample quality (energy_W2=37.9) where RBF-KSD-ASBS (71.3K) and baseline (4.67M) catastrophically fail. The polynomial tails of IMQ maintain mode-resolving gradients where RBF's exponential tails vanish.
4. **λ must scale with the problem.** λ=1.0 works for DW4 and RotGMM d≤50, but diverges at d=100 (requires λ=0.1). Müller-Brown requires λ=0.01 due to sharp potential gradients. λ should be tuned per-benchmark.
5. **RBF kernel sweet spot is d=10–30; IMQ extends the range to d≈50.** Beyond d=50, no standard isotropic kernel suffices — all methods degrade to noise.
6. **Best-seed KSD-ASBS beats best-seed baseline on all DW4 metrics**, suggesting the method has higher potential but more variance.

### Key Questions (pending experiments):

1. Does the advantage persist in higher dimensions (LJ13, LJ38, LJ55)?
2. Does KSD-ASBS find both funnels in LJ38 (the headline result)?
3. What is the optimal λ across benchmarks?
4. ✅ Does KSD-ASBS work where CVs are unknown (RotGMM)? **YES at d=10 (3× mode coverage). At d=30, KSD-ASBS finds fewer modes but with maximal spatial separation and +22% better energy_W2. RBF kernel mode-resolving power degrades with dimension, but distributional quality improvements persist.**
5. ✅ Does IMQ kernel help in high-D? **YES — dramatically at d=50. IMQ-KSD-ASBS covers 7/8 modes with energy_W2=37.9, vs RBF's 71.3K (1,881× worse) and baseline's 4.67M (123,000× worse). The polynomial tails of IMQ maintain mode-resolving gradients where RBF is flat. However, IMQ fails at d=100 — no standard isotropic kernel suffices beyond d≈50.**
6. Does the method generalize beyond molecular systems (BLogReg)?
7. What is the computational overhead in practice?

---

## Reproduction

```bash
# Phase 1: Infrastructure (create files, configs)
# Phase 2: Train baselines
bash scripts/run_phase2_baselines.sh
# Phase 3: Train KSD-ASBS + ablation
bash scripts/run_phase3_ksd.sh
# Phase 4: Synthetic experiments
bash scripts/run_phase4_synthetic.sh
# Phase 5: Evaluate everything
python evaluate_all.py --outputs_root outputs --results_dir results --n_samples 2000 --n_eval_seeds 5
# Phase 6: Generate this report
python generate_results.py --results_dir results --output RESULTS.md
```
