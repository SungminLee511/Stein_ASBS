# Stein_ASBS — SDR-Augmented Adjoint Schrödinger Bridge Sampler

## Overview
Forked from [facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_samplers). Research project that augments Meta's ASBS with **SDR (Stein Discrepancy Regularization)** — combining a KSD penalty with DARW reweighting to prevent mode collapse in sampling from Boltzmann distributions.

**Core Idea:** SDR adds two modifications to ASBS: (1) an inter-particle KSD term to the adjoint terminal condition, and (2) DARW importance reweighting of the AM regression loss. Everything else stays identical.

**Key equation (the only change to ASBS):**
```
Y₁ⁱ = -(1/N)∇Φ₀(X₁ⁱ) - (λ/N²) Σⱼ ∇ₓ kₚ(X₁ⁱ, X₁ʲ)
       ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       standard adjoint         SDR correction (NEW)
```

## File Tree

```
Stein_ASBS/
├── .claude/
│   ├── skills/
│   │   └── SKILL.md              # This file — project conventions & architecture
│   └── TODO/
│       └── *.md                  # Implementation guides and notes
├── train.py                      # Entry point — Hydra-based training script
├── adjoint_samplers/             # Main package
│   ├── train_loop.py             # Training loop logic
│   ├── components/
│   │   ├── matcher.py            # AdjointVEMatcher, AdjointVPMatcher (base)
│   │   ├── sdr_matcher.py        # SDRAdjointVE/VPMatcher (SDR)
│   │   ├── stein_kernel.py       # KSD computation, Stein kernel gradient
│   │   ├── generic_evaluator.py  # GenericEnergyEvaluator, RotatedGMMEvaluator
│   │   ├── buffer.py             # BatchBuffer for trajectory data
│   │   ├── evaluator.py          # SyntheticEnergyEvaluator (base)
│   │   ├── model.py              # FourierMLP, EGNN architectures
│   │   ├── sde.py                # SDE definitions, sdeint, ControlledSDE
│   │   ├── state_cost.py         # ZeroGradStateCost
│   │   └── term_cost.py          # Terminal cost (score/corrector)
│   ├── energies/
│   │   ├── base_energy.py        # Abstract energy interface
│   │   ├── double_well_energy.py # DW4 benchmark (8D)
│   │   ├── lennard_jones_energy.py # LJ13/LJ38/LJ55 benchmarks
│   │   ├── dist_energy.py        # Distribution-based energy
│   │   ├── rotated_gmm_energy.py # RotatedGMMEnergy
│   │   ├── muller_brown_energy.py # MullerBrownEnergy (2D)
│   │   ├── bayesian_logreg_energy.py # BayesianLogRegEnergy
│   │   ├── aldp_energy.py        # AlanineDipeptideEnergy (22 atoms, 66D, OpenMM+Amber14)
│   │   ├── viz_energies.py       # GMM9Energy, Grid25Energy, Ring8Energy, BananaEnergy
│   │   └── new_benchmarks.py     # UnequalGMMEnergy, ManyWell5DEnergy, ManyWell32DEnergy
│   └── utils/
│       ├── train_utils.py        # get_timesteps, training helpers
│       ├── eval_utils.py         # interatomic_dist, dist_point_clouds
│       ├── graph_utils.py        # Graph/COM-free coordinate helpers
│       ├── dist_utils.py         # Distributed training utils
│       └── distributed_mode.py   # Multi-GPU setup
├── configs/                      # Hydra config hierarchy
│   ├── train.yaml                # Top-level training config
│   ├── experiment/               # Active experiment configs:
│   │   ├── dw4_as.yaml           # DW4 AS baseline
│   │   ├── dw4_asbs.yaml         # DW4 ASBS baseline
│   │   ├── grid25_as.yaml        # Grid25 AS baseline
│   │   ├── grid25_asbs.yaml      # Grid25 ASBS baseline
│   │   ├── grid25_sdr_asbs.yaml # Grid25 SDR
│   │   ├── mw5_asbs.yaml         # MW5 ASBS baseline
│   │   ├── mw32_asbs.yaml        # MW32 ASBS baseline
│   │   ├── mw32_sdr_asbs.yaml   # MW32 SDR
│   │   ├── lj13_as.yaml          # LJ13 AS baseline
│   │   ├── lj13_asbs.yaml        # LJ13 ASBS baseline
│   │   ├── lj13_sdr_asbs.yaml   # LJ13 SDR
│   │   ├── lj38_asbs.yaml        # LJ38 ASBS baseline
│   │   ├── lj38_sdr_asbs.yaml   # LJ38 SDR
│   │   ├── lj55_as.yaml          # LJ55 AS baseline
│   │   ├── lj55_asbs.yaml        # LJ55 ASBS baseline
│   │   ├── lj55_sdr_asbs.yaml   # LJ55 SDR
│   │   ├── aldp_asbs.yaml        # ALDP ASBS baseline (skip_eval until ref data ready)
│   │   └── aldp_sdr_asbs.yaml   # ALDP SDR (skip_eval until ref data ready)
│   ├── problem/                  # dw4, grid25, mw5, mw32, lj13, lj38, lj55, aldp
│   ├── matcher/                  # adjoint_ve/vp, sdr_adjoint_ve, sdr_adjoint_ve, imq variants
│   ├── sde/                      # ve, vp, brownian_motion, graph_ve, graph_vp
│   ├── source/                   # gauss, harmonic, delta, meanfree
│   ├── model/                    # fouriermlp, egnn
│   ├── state_cost/               # zero
│   ├── term_cost/                # score_term_cost, corrector_term_cost, graph variants
│   └── lancher/                  # Slurm launcher config
├── scripts/
│   ├── eval_grid25_sdr.py       # Grid25 SDR evaluation (metrics + marginal evolution figure)
│   ├── run_grid25_darw.sh        # Grid25 SDR launcher (3 betas × 3 seeds)
│   └── run_12_darw_parallel.sh   # 12-experiment parallel launcher
├── evaluation/
│   ├── RESULT.md                 # Grid25 evaluation results summary
│   ├── grid25_darw_results.json  # Raw metrics JSON
│   └── figures_2d/               # NeurIPS-style marginal evolution figures
├── BASELINE_MODEL/               # Baseline models for comparison
│   ├── dem/                      # DEM (Diffusion Energy Matching) codebase
│   ├── dw4_asbs/                 # DW4 ASBS baseline results
│   └── lj13_asbs/                # LJ13 ASBS baseline results
├── PORTAL/                       # (empty — reserved for publication artifacts)
├── results/                      # Training outputs (gitignored)
├── data/                         # Reference test splits (test_split_*.npy), alanine-dipeptide.pdb
├── PLAN.md                       # Experiment execution plan
├── environment.yml               # Conda environment spec
├── README.md, LICENSE.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md
└── assets/demo.png
```

## Architecture

### ASBS Training Pipeline
1. **Sample** N particles from source distribution μ
2. **Forward SDE** simulate each particle: `sdeint(sde, x0, timesteps)`
3. **Adjoint terminal condition** compute Y₁ⁱ from terminal cost gradient
4. **Backward** propagate adjoint (VP-SDE) or keep constant (VE-SDE)
5. **DARW weights** (optional) compute per-sample importance weights from p̃/q̂ ratio
6. **Buffer** store (t, xₜ, Yₜ, weights) tuples
7. **AM Regression** train controller uθ(x,t) against -Yₜ via weighted MSE (weights from DARW)

### Key Components

| Component | File | Role |
|-----------|------|------|
| `AdjointVEMatcher` | `components/matcher.py` | VE adjoint matching (base class) |
| `AdjointVPMatcher` | `components/matcher.py` | VP adjoint matching (base class) |
| `SDRAdjointVEMatcher` | `components/sdr_matcher.py` | Inherits VE, adds SDR (KSD + DARW) to `populate_buffer` |
| `SDRAdjointVPMatcher` | `components/sdr_matcher.py` | Inherits VP, adds SDR (KSD + DARW) to `populate_buffer` |
| `stein_kernel.py` | `components/stein_kernel.py` | Standalone KSD/gradient compute |
| `ControlledSDE` | `components/sde.py` | Wraps ref_sde + controller into controlled SDE |
| `BatchBuffer` | `components/buffer.py` | Stores trajectory samples for AM regression |
| `GenericEnergyEvaluator` | `components/generic_evaluator.py` | Energy W2 + Sinkhorn eval |
| `FourierMLP` | `components/model.py` | MLP with Fourier time embedding |
| `EGNN` | `components/model.py` | Equivariant GNN for graph problems (DW4/LJ) |

### Active Benchmarks

| Benchmark | Dim | Energy | Config prefix |
|-----------|-----|--------|---------------|
| DW4 | 8 (4×2D) | Double well | `dw4_` |
| LJ13 | 39 (13×3D) | Lennard-Jones | `lj13_` |
| LJ38 | 114 (38×3D) | Lennard-Jones | `lj38_` |
| LJ55 | 165 (55×3D) | Lennard-Jones | `lj55_` |
| Grid25 | 2 | 25-mode GMM (5×5 grid) | `grid25_` |
| MW5 | 5 | 5D double-well (32 modes) | `mw5_` |

### Hyperparameters (SDR)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `sdr_lambda` | 1.0 | SDR penalty weight |
| `sdr_bandwidth` | null (median heuristic) | Kernel bandwidth |
| `sdr_kernel` | "rbf" | "rbf" or "imq" |
| `sdr_beta` | 0.0 (disabled) | SDR DARW reweighting strength. Ablate: {0.3, 0.5, 0.7, 1.0} |
| `sdr_weight_clip` | 10.0 | Max DARW weight before normalization |
| `clip_grad_norm` | false (MW5: 1.0) | Model gradient norm clipping. Accepts any float. |
| `max_grad_E_norm` | 50 | Score/force clipping norm before Stein kernel. MW5 and Grid25 both use 50. |
| `sigma_min` | 0.01 | Noise floor for SDE. MW5 changed from 0.001 to 0.01 for stability. |

## Multi-Server Setup
- **This server**: Grid25 and MW5 experiments (2D/5D)
- **Other server**: LJ and DW4 experiments (molecular)
- Conda env: `adjoint_samplers`
- Run: `PYTHONPATH=/home/sky/SML/Stein_ASBS python train.py experiment=<name> ...`

## Gotchas
1. **DO NOT modify existing base files** (`matcher.py`, `evaluator.py`, etc.). New code inherits/extends only. Exceptions: `train_loop.py` (NaN-safety + DARW weighted loss), `base_energy.py` (added `"energy"` key to `__call__`).
2. **Device consistency** — all tensors in `stein_kernel.py` must stay on same device (CUDA).
3. **Gradient detaching** — `_apply_sdr_correction` runs under `@torch.no_grad()`. SDR correction modifies the adjoint *target*, NOT the loss.
4. **SDR backward compat** — with `sdr_beta: 0`, SDR DARW is disabled and all weights are 1.0.
5. **`prepare_target` return signature** — SDR matchers return 3 values `(input, target, weights)`. Base matchers return 2 values. `train_loop.py` handles both via `len(result)` check.
6. **`results/` is gitignored** — checkpoints are local only.
7. **NaN prevention (MW5)** — MW5 5D energy has extreme gradients (±72.5/dim). Multiple safeguards added: (a) Gamma clamped to ±1e4 in `_stein_grad_rbf`, (b) DARW weights computed in log-space with clamped log-ratios, (c) `nan_to_num` on Stein grad output and DARW weights, (d) `clip_grad_norm: 1.0` default in MW5 config.
8. **`clip_grad_norm` accepts any float** — previously values <1 were silently overridden to 1.0 (bug fixed).
