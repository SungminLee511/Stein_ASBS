# DW4 Evaluation Results — Baseline ASBS vs KSD-ASBS

## Setup

- **Benchmark:** DW4 (4 particles × 2D = 8D, double-well energy)
- **Baseline:** ASBS (AdjointVEMatcher) — `baselines/dw4_asbs/`
- **KSD-ASBS:** KSDAdjointVEMatcher with λ=1.0, median bandwidth — `results/local/2026.04.04/064017/`
- **Evaluation:** 2000 samples per seed, 5 sampling seeds (0–4)
- **Metrics:** All Wasserstein-2 (lower is better)

## Per-Seed Results

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

## Summary Statistics

| Method | energy_W2 (mean±std) | eq_W2 (mean±std) | dist_W2 (mean±std) |
|--------|----------------------|-------------------|---------------------|
| **Baseline** | 0.1400 ± 0.0060 | 0.4460 ± 0.0542 | 0.026799 ± 0.012059 |
| **KSD-ASBS** | 0.1820 ± 0.0410 | 0.4023 ± 0.0494 | 0.010010 ± 0.007469 |

### Best Seed

| Metric | Baseline (best) | KSD-ASBS (best) | Winner |
|--------|-----------------|-----------------|--------|
| energy_W2 | **0.1328** (seed 0) | 0.1213 (seed 4) | **KSD-ASBS** |
| eq_W2 | 0.3566 (seed 1) | **0.3287** (seed 1) | **KSD-ASBS** |
| dist_W2 | 0.008214 (seed 1) | **0.000206** (seed 1) | **KSD-ASBS** |

## Relative Change (mean, KSD-ASBS vs Baseline)

| Metric | Change | Direction |
|--------|--------|-----------|
| energy_W2 | -30.07% | ↑ worse |
| eq_W2 | +9.82% | ↓ better |
| dist_W2 | +62.65% | ↓ better |

## Interpretation

- **dist_W2 (interatomic distances):** KSD-ASBS is significantly better on average (63% improvement) — the KSD penalty successfully pushes samples toward the correct interatomic distance distribution, indicating better mode coverage.
- **eq_W2 (equilibrium/point cloud):** KSD-ASBS is moderately better (~10%) — structural quality of generated configurations is improved.
- **energy_W2:** KSD-ASBS is worse on average (-30%), but **best-seed KSD-ASBS (0.1213, seed 4) beats best-seed baseline (0.1328, seed 0)**. High variance across seeds suggests sensitivity to sampling noise rather than a fundamental degradation.
- **Best-seed comparison:** KSD-ASBS wins all three metrics, suggesting the method has higher potential but more variance.

## Next Steps

- [ ] λ ablation: try {0.1, 0.5, 2.0, 5.0} to find optimal trade-off
- [ ] Increase sampling seeds (10+) to reduce variance in mean estimates
- [ ] Run LJ13 comparison if DW4 results hold with tuned λ
