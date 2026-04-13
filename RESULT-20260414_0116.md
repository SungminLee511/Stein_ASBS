# Grid25 SDR Evaluation Results

**Benchmark:** Grid25 (5x5 Gaussian mixture, 2D)
**Methods:** Vanilla ASBS vs SDR-ASBS with beta = {0.5, 0.7, 1.0}
**Seeds:** 3 per method (mean +/- std)
**Samples:** 2000 per evaluation
**Excluded metrics:** ESS, ESS %, Std Energy (not reported)

## Results Table

| Metric | ASBS | SDR beta=0.5 | SDR beta=0.7 | SDR beta=1.0 |
|---|---|---|---|---|
| Mode Weight TV (lower is better) | 0.254 +/- 0.075 | 0.162 +/- 0.036 | **0.095 +/- 0.015** | 0.122 +/- 0.041 |
| Energy W2 (lower is better) | **0.135 +/- 0.032** | 0.223 +/- 0.021 | 0.326 +/- 0.027 | 0.537 +/- 0.147 |
| W2 Distance (lower is better) | 1.768 +/- 0.287 | 1.273 +/- 0.299 | **0.737 +/- 0.151** | 0.876 +/- 0.361 |
| Sinkhorn Divergence (lower is better) | 3.140 +/- 0.959 | 1.752 +/- 0.770 | **0.631 +/- 0.224** | 0.962 +/- 0.689 |
| KL Divergence (lower is better) | 2.406 +/- 0.465 | **2.178 +/- 0.079** | 2.227 +/- 0.083 | 2.259 +/- 0.102 |

## Key Findings

- **SDR beta=0.7 is the best overall**, winning Mode Weight TV, W2 Distance, and Sinkhorn Divergence with lowest variance.
- **All SDR variants improve mode coverage** over vanilla ASBS (lower Mode Weight TV).
- **Trade-off exists**: higher beta improves distributional quality (W2, Sinkhorn) but degrades energy accuracy (Energy W2).
- **KL Divergence** is similar across all methods, with SDR beta=0.5 slightly best.
- **SDR beta=1.0** is competitive but has higher variance than beta=0.7.

## Experiment Details

- **Grid25 SDR beta=1.0**: 3000 epochs, clip_grad_norm=1.0, lr=1e-3
- **Grid25 SDR beta=0.5, 0.7**: 3000 epochs (DARW variant), default settings
- **ASBS baseline**: vanilla adjoint sampler without SDR
- **Evaluation date**: 2026-04-14
