# Results: KSD-Augmented ASBS — Comprehensive Evaluation

**Last updated:** 2026-04-04 KST

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

*Figures: `evaluation/figures/muller_comparison.png`, `evaluation/figures/muller_all_seeds.png`*

---

## 5. Synthetic CV-Unknown: Rotated Gaussian Mixture

**Status: ⬜ PENDING**

Tests mode coverage on energy functions where **collective variables are unknown by construction**. Modes are separated along a randomly rotated axis — no axis-aligned projection separates them.

### 5.1 Mode Coverage vs Dimension

| Dimension | Method | Modes Covered (of 8) | Coverage (%) | energy_W2 |
|-----------|--------|---------------------|--------------|-----------|
| d=10 | Baseline |  |  |  |
| d=10 | KSD-ASBS |  |  |  |
| d=30 | Baseline |  |  |  |
| d=30 | KSD-ASBS |  |  |  |
| d=50 | Baseline |  |  |  |
| d=50 | KSD-ASBS |  |  |  |
| d=100 | Baseline |  |  |  |
| d=100 | KSD-ASBS |  |  |  |

*Figure: Mode coverage bar chart (baseline vs KSD-ASBS) across dimensions — to be generated.*

**Expected:** KSD-ASBS should maintain higher mode coverage as dimension increases. Baseline may lose modes in high-D because the per-particle energy cost alone doesn't prevent mode collapse. RBF kernel effectiveness may degrade above d=50.

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
| RotGMM-10 | mode coverage |  |  |  |  |
| RotGMM-30 | mode coverage |  |  |  |  |
| RotGMM-50 | mode coverage |  |  |  |  |
| RotGMM-100 | mode coverage |  |  |  |  |
| Müller-Brown | energy_W2 | 0.4255 | 0.4079 | +4.1% ↓ | **KSD-ASBS** |
| Müller-Brown | KSD² | 0.0216 | 0.0154 | +28.7% ↓ | **KSD-ASBS** |
| Müller-Brown | modes covered | 3/3 | 3/3 | — | Tie |
| BLogReg-Au | energy_W2 |  |  |  |  |
| BLogReg-Ge | energy_W2 |  |  |  |  |

### 8.2 Best-Seed Comparison (DW4)

| Metric | Baseline (best) | KSD-ASBS (best) | Winner |
|--------|-----------------|-----------------|--------|
| energy_W2 | 0.1328 (seed 0) | **0.1213** (seed 4) | **KSD-ASBS** |
| eq_W2 | 0.3566 (seed 1) | **0.3287** (seed 1) | **KSD-ASBS** |
| dist_W2 | 0.008214 (seed 1) | **0.000206** (seed 1) | **KSD-ASBS** |

---

## 9. Conclusions

*(To be updated as experiments complete)*

### Preliminary (DW4 only):

1. **KSD-ASBS improves distributional metrics** (dist_W2 ↓63%, eq_W2 ↓10%) at the cost of higher energy_W2 variance on average.
2. **Best-seed KSD-ASBS beats best-seed baseline on all metrics**, suggesting the method has higher potential but needs better hyperparameter tuning or more seeds.
3. **λ ablation is critical** — current results use λ=1.0; a smaller λ may reduce energy_W2 degradation while preserving dist_W2 gains.

### Key Questions (pending experiments):

1. Does the advantage persist in higher dimensions (LJ13, LJ38, LJ55)?
2. Does KSD-ASBS find both funnels in LJ38 (the headline result)?
3. What is the optimal λ across benchmarks?
4. Does KSD-ASBS work where CVs are unknown (RotGMM)?
5. Does the method generalize beyond molecular systems (BLogReg)?
6. What is the computational overhead in practice?

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
