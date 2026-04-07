# 2D Visualization Benchmark Results

Generated: 2026-04-07 08:29:33 KST

---

## 9-Mode GMM (3x3 Grid)

| Metric | ASBS (Baseline) | KSD-ASBS (lambda=0.01) |
|---|---|---|
| Modes covered (of 9) | 5 | 8 |
| Mean energy | 1.2632 | 1.0165 |
| Std energy | 1.2876 | 1.0296 |
| Per-mode counts | [0, 7, 213, 30, 99, 1595, 0, 0, 0] | [185, 223, 152, 260, 170, 344, 364, 275, 0] |

### Terminal Distribution

![gmm9 terminal](figures_2d/gmm9_terminal.png)

### Marginal Evolution: ASBS

![gmm9 marginal asbs](figures_2d/gmm9_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![gmm9 marginal ksd](figures_2d/gmm9_marginal_ksd.png)

---

## 9-Mode GMM (3x3 Grid) — Seed 1

Evaluated: 2026-04-07 09:06:12 KST | Checkpoint: latest (~ep 2316, early-stopped)

| Metric | ASBS (Baseline) | KSD-ASBS (lambda=0.01) |
|---|---|---|
| Modes covered (of 9) | 9 | 9 |
| Mean energy | 1.0573 | 1.0621 |
| Std energy | 1.0890 | 1.2858 |
| Per-mode counts | [60, 110, 1066, 25, 35, 253, 44, 73, 302] | [341, 317, 697, 94, 98, 191, 71, 62, 99] |

### Terminal Distribution

![gmm9_s1 terminal](figures_2d/gmm9_s1_terminal.png)

### Marginal Evolution: ASBS

![gmm9_s1 marginal asbs](figures_2d/gmm9_s1_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![gmm9_s1 marginal ksd](figures_2d/gmm9_s1_marginal_ksd.png)

---

## Two Moons

| Metric | ASBS (Baseline) | KSD-ASBS (lambda=0.01) |
|---|---|---|
| Modes covered (of 2) | 2 | 2 |
| Mean energy | 0.1167 | 0.0779 |
| Std energy | 0.7577 | 0.6791 |
| Per-mode counts | [895, 1102] | [1175, 821] |

### Terminal Distribution

![two_moons terminal](figures_2d/two_moons_terminal.png)

### Marginal Evolution: ASBS

![two_moons marginal asbs](figures_2d/two_moons_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![two_moons marginal ksd](figures_2d/two_moons_marginal_ksd.png)

---

## Checkerboard (4x4)

| Metric | ASBS (Baseline) | KSD-ASBS (lambda=0.1) |
|---|---|---|
| Modes covered (of 8) | 8 | 8 |
| Mean energy | 0.8375 | 0.8239 |
| Std energy | 1.1476 | 1.1650 |
| Per-mode counts | [64, 133, 95, 103, 70, 103, 76, 73] | [74, 107, 85, 87, 92, 97, 95, 78] |

### Terminal Distribution

![checkerboard terminal](figures_2d/checkerboard_terminal.png)

### Marginal Evolution: ASBS

![checkerboard marginal asbs](figures_2d/checkerboard_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![checkerboard marginal ksd](figures_2d/checkerboard_marginal_ksd.png)

---

## Spiral

| Metric | ASBS (Baseline) | KSD-ASBS (lambda=0.01) |
|---|---|---|
| Modes covered (of 20) | 19 | 20 |
| Mean energy | 0.5350 | 0.5466 |
| Std energy | 0.7754 | 0.8608 |
| Per-mode counts | [6, 13, 13, 8, 24, 21, 19, 19, 5, 7, 5, 7, 31, 98, 165, 149, 136, 123, 19, 0] | [20, 31, 25, 41, 49, 48, 39, 36, 30, 82, 119, 128, 130, 71, 62, 68, 53, 34, 34, 54] |

### Terminal Distribution

![spiral terminal](figures_2d/spiral_terminal.png)

### Marginal Evolution: ASBS

![spiral marginal asbs](figures_2d/spiral_marginal_asbs.png)

### Marginal Evolution: KSD-ASBS

![spiral marginal ksd](figures_2d/spiral_marginal_ksd.png)

---
## 25-Mode Grid (5×5)

| Metric | ASBS (Baseline) | SDR-ASBS (λ=0.1) |
|---|---|---|
| Modes covered (of 25) | 25 | 25 |
| Mean energy | 1.0373 | 1.0327 |
| Std energy | 1.0660 | 1.0167 |
| KL divergence | 2.0499 | 2.1956 |
| W₂ distance | 2.1668 | 1.1025 |
| Sinkhorn divergence | 3.4123 | 1.2662 |
| Mode weight TV | 0.3035 | 0.1503 |
| ESS | 3.8 (0.19%) | 3.7 (0.19%) |
| Per-mode counts | [13, 17, 53, 93, 159, 20, 25, 64, 113, 164, 20, 42, 67, 140, 173, 17, 34, 70, 104, 135, 18, 43, 85, 126, 166] | [61, 69, 106, 143, 227, 65, 53, 61, 84, 75, 49, 50, 65, 98, 71, 62, 67, 71, 65, 47, 74, 89, 81, 98, 38] |

### Terminal Distribution

![grid25 terminal](figures_2d/grid25_terminal_neurips.png)

### Marginal Evolution: ASBS

![grid25 marginal asbs](figures_2d/grid25_marginal_asbs_neurips.png)

### Marginal Evolution: SDR-ASBS

![grid25 marginal sdr](figures_2d/grid25_marginal_sdr_neurips.png)

---

