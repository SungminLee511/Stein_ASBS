# SDR → Vanilla ASBS Fine-tuning Results

## Hypothesis
Fine-tuning SDR-ASBS checkpoints with sdr_lambda=0 recovers Energy W2
while retaining SDR's mode coverage advantages.

## Baselines (from main evaluation)

### Grid25
| Metric | ASBS | SDR beta=0.7 |
|---|---|---|
| Mode Weight TV ↓ | 0.254 +/- 0.075 | **0.095 +/- 0.015** |
| Energy W2 ↓ | **0.135 +/- 0.032** | 0.326 +/- 0.027 |
| W2 Distance ↓ | 1.768 +/- 0.287 | **0.737 +/- 0.151** |
| Sinkhorn ↓ | (debiased, pending) | (debiased, pending) |

### MW5
| Metric | ASBS | SDR beta=0.7 |
|---|---|---|
| Mode Weight TV ↓ | **0.325 +/- 0.087** | 0.668 +/- 0.036 |
| Energy W2 ↓ | (pending) | (pending) |
| W2 Distance ↓ | 3.247 +/- 0.474 | **1.198 +/- 0.115** |
| Sinkhorn ↓ | (debiased, pending) | (debiased, pending) |

---

## Fine-tune Results

### Grid25 (SDR beta=0.7 → lambda=0, LR=5e-4)

*TODO: fill after running experiments*

| Fine-tune epochs | Mode TV ↓ | Energy W2 ↓ | W2 ↓ | Sinkhorn ↓ |
|---|---|---|---|---|
| +50 | | | | |
| +100 | | | | |
| +200 | | | | |
| +500 | | | | |

### MW5 (SDR beta=0.7 → lambda=0, LR=5e-4)

*TODO: fill after running experiments*

| Fine-tune epochs | Mode TV ↓ | Energy W2 ↓ | W2 ↓ | Sinkhorn ↓ |
|---|---|---|---|---|
| +50 | | | | |
| +100 | | | | |
| +200 | | | | |
| +500 | | | | |
