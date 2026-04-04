# Stein_ASBS — KSD-Augmented Adjoint Schrödinger Bridge Sampler

## Overview
Forked from [facebookresearch/adjoint_samplers](https://github.com/facebookresearch/adjoint_samplers). Research project that augments Meta's ASBS (Adjoint Schrödinger Bridge Sampler) with a **Kernel Stein Discrepancy (KSD)** penalty to prevent mode collapse in sampling from Boltzmann distributions.

**Core Idea:** Add an inter-particle KSD term to the terminal cost of the stochastic optimal control (SOC) problem. This modifies only the adjoint terminal condition — everything else (SDE integration, backward sim, buffer, AM regression, training loop) stays identical.

**Key equation (the only change to ASBS):**
```
Y₁ⁱ = -(1/N)∇Φ₀(X₁ⁱ) - (λ/N²) Σⱼ ∇ₓ kₚ(X₁ⁱ, X₁ʲ)
       ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       standard adjoint         KSD correction (NEW)
```

## File Tree

```
Stein_ASBS/
├── .claude/
│   ├── skills/
│   │   └── SKILL.md              # This file — project conventions & architecture
│   └── TODO/
│       └── implementation_guide.md  # Step-by-step build plan with full code
├── train.py                      # Entry point — Hydra-based training script (DO NOT MODIFY)
├── adjoint_samplers/             # Main package
│   ├── train_loop.py             # Training loop logic (DO NOT MODIFY)
│   ├── components/
│   │   ├── matcher.py            # EXISTING — AdjointVEMatcher, AdjointVPMatcher (DO NOT MODIFY)
│   │   ├── buffer.py             # EXISTING — BatchBuffer for trajectory data
│   │   ├── evaluator.py          # EXISTING — SyntheticEnergyEvaluator
│   │   ├── generic_evaluator.py  # NEW — GenericEnergyEvaluator, RotatedGMMEvaluator
│   │   ├── model.py              # EXISTING — FourierMLP, EGNN architectures
│   │   ├── sde.py                # EXISTING — SDE definitions, sdeint, ControlledSDE
│   │   ├── state_cost.py         # EXISTING — ZeroGradStateCost
│   │   ├── term_cost.py          # EXISTING — terminal cost (score/corrector)
│   │   ├── stein_kernel.py       # NEW — KSD computation, Stein kernel gradient
│   │   └── ksd_matcher.py        # NEW — KSDAdjointVE/VPMatcher
│   ├── energies/
│   │   ├── base_energy.py        # Abstract energy interface
│   │   ├── double_well_energy.py # DW4 benchmark (8D, 4 particles × 2D)
│   │   ├── lennard_jones_energy.py # LJ13/LJ38/LJ55 benchmarks
│   │   ├── dist_energy.py        # Distribution-based energy
│   │   ├── rotated_gmm_energy.py # NEW — RotatedGMMEnergy (synthetic CV-unknown)
│   │   ├── muller_brown_energy.py # NEW — MullerBrownEnergy (2D visualization)
│   │   └── bayesian_logreg_energy.py # NEW — BayesianLogRegEnergy (non-molecular)
│   └── utils/
│       ├── train_utils.py        # get_timesteps, training helpers
│       ├── eval_utils.py         # interatomic_dist, dist_point_clouds
│       ├── graph_utils.py        # Graph/COM-free coordinate helpers
│       ├── dist_utils.py         # Distributed training utils
│       └── distributed_mode.py   # Multi-GPU setup
├── configs/                      # Hydra config hierarchy
│   ├── train.yaml                # Top-level training config
│   ├── experiment/
│   │   ├── dw4_asbs.yaml         # EXISTING — DW4 baseline
│   │   ├── dw4_ksd_asbs.yaml     # NEW — DW4 + KSD
│   │   ├── lj13_asbs.yaml        # EXISTING — LJ13 baseline
│   │   ├── lj13_ksd_asbs.yaml    # NEW — LJ13 + KSD
│   │   ├── lj38_asbs.yaml        # NEW — LJ38 baseline (double-funnel)
│   │   ├── lj38_ksd_asbs.yaml    # NEW — LJ38 + KSD
│   │   ├── lj55_asbs.yaml        # EXISTING — LJ55 baseline
│   │   ├── lj55_ksd_asbs.yaml    # NEW — LJ55 + KSD
│   │   ├── muller_asbs.yaml      # NEW — Müller-Brown baseline
│   │   ├── muller_ksd_asbs.yaml  # NEW — Müller-Brown + KSD
│   │   ├── blogreg_au_asbs.yaml  # NEW — Bayesian LogReg Australian baseline
│   │   ├── blogreg_au_ksd_asbs.yaml # NEW — Australian + KSD
│   │   ├── blogreg_ge_asbs.yaml  # NEW — Bayesian LogReg German baseline
│   │   ├── blogreg_ge_ksd_asbs.yaml # NEW — German + KSD
│   │   ├── rotgmm10_asbs.yaml    # NEW — RotGMM d=10 baseline
│   │   ├── rotgmm10_ksd_asbs.yaml # NEW — RotGMM d=10 + KSD
│   │   ├── rotgmm30_asbs.yaml    # NEW — RotGMM d=30 baseline
│   │   ├── rotgmm30_ksd_asbs.yaml
│   │   ├── rotgmm50_asbs.yaml    # NEW — RotGMM d=50 baseline
│   │   ├── rotgmm50_ksd_asbs.yaml
│   │   ├── rotgmm100_asbs.yaml   # NEW — RotGMM d=100 baseline
│   │   └── rotgmm100_ksd_asbs.yaml
│   ├── matcher/
│   │   ├── adjoint_ve.yaml       # EXISTING — VE adjoint matcher
│   │   ├── adjoint_vp.yaml       # EXISTING — VP adjoint matcher
│   │   ├── corrector.yaml        # EXISTING — corrector matcher
│   │   └── ksd_adjoint_ve.yaml   # NEW — KSD VE matcher
│   ├── sde/                      # ve.yaml, vp.yaml, graph_ve.yaml, etc.
│   ├── problem/                  # dw4, lj13, lj38, lj55, muller, blogreg_*, rotgmm*
│   ├── source/                   # gauss.yaml, harmonic.yaml, delta.yaml, meanfree.yaml
│   ├── model/                    # fouriermlp.yaml, egnn.yaml
│   ├── state_cost/               # zero.yaml
│   ├── term_cost/                # score_term_cost.yaml, corrector_term_cost.yaml, graph_*
│   └── lancher/                  # Slurm launcher config
├── scripts/
│   ├── dw4.sh                    # DW4 training script (original)
│   ├── lj13.sh                   # LJ13 training script (original)
│   ├── lj55.sh                   # LJ55 training script (original)
│   ├── demo.sh                   # Demo script (original)
│   ├── download.sh               # Download reference test samples
│   ├── run_phase2_baselines.sh   # NEW — all baseline training (Phase 2)
│   ├── run_phase3_ksd.sh         # NEW — KSD training + λ ablation (Phase 3)
│   ├── run_phase4_synthetic.sh   # NEW — RotGMM experiments (Phase 4)
│   ├── run_phase4b_cvunknown.sh  # NEW — Müller-Brown experiments (Phase 4b)
│   ├── run_phase4c_nonmolecular.sh # NEW — BLogReg experiments (Phase 4c)
│   └── run_phase5_evaluate.sh    # NEW — full evaluation + report (Phase 5)
├── evaluation/                   # All evaluation-related files
│   ├── evaluate_comparison.py    # DW4 baseline vs KSD comparison (early eval)
│   ├── evaluate_all.py           # Master evaluation script (Phase 5)
│   ├── generate_results.py       # Auto-generate RESULTS.md (Phase 6)
│   ├── run_phase5_evaluate.sh    # Evaluation run script
│   ├── RESULTS.md                # Results (auto-generated + manually edited)
│   ├── eval_results_dw4.json     # DW4 eval metrics
│   └── eval_comparison_log.txt   # DW4 comparison log
├── PLAN.md                       # Experiment execution plan
├── environment.yml               # Conda environment spec
├── LICENSE.md                    # Meta license
└── CONTRIBUTING.md
```

## Architecture

### ASBS Training Pipeline
1. **Sample** N particles from source distribution μ
2. **Forward SDE** simulate each particle: `sdeint(sde, x0, timesteps)`
3. **Adjoint terminal condition** compute Y₁ⁱ from terminal cost gradient
4. **Backward** propagate adjoint (VP-SDE) or keep constant (VE-SDE)
5. **Buffer** store (t, xₜ, Yₜ) tuples
6. **AM Regression** train controller uθ(x,t) against -Yₜ via MSE

### KSD Modification (Steps 3–4 only)
- After computing standard adjoint `a₀ⁱ = -∇Φ₀(x₁ⁱ)`, add KSD correction:
  `Δᵢ = (λ/N²) Σⱼ ∇ₓ kₚ(x₁ⁱ, x₁ʲ)` using detached Stein kernel gradient
- Augmented adjoint: `Y₁ⁱ = (1/N)·a₀ⁱ + Δᵢ`
- Uses Hessian-free approximation (detach scores from graph): O(N²d) cost

### Key Components

| Component | File | Role |
|-----------|------|------|
| `AdjointVEMatcher` | `components/matcher.py` | VE adjoint matching (base class) |
| `AdjointVPMatcher` | `components/matcher.py` | VP adjoint matching (base class) |
| `KSDAdjointVEMatcher` | `components/ksd_matcher.py` | **NEW** — inherits VE, adds KSD to `populate_buffer` |
| `KSDAdjointVPMatcher` | `components/ksd_matcher.py` | **NEW** — inherits VP, adds KSD to `populate_buffer` |
| `stein_kernel.py` | `components/stein_kernel.py` | **NEW** — standalone KSD/gradient compute |
| `ControlledSDE` | `components/sde.py` | Wraps ref_sde + controller into controlled SDE |
| `BatchBuffer` | `components/buffer.py` | Stores trajectory samples for AM regression |
| `EGNN` | `components/model.py` | Equivariant GNN for graph problems (DW4/LJ) |
| `FourierMLP` | `components/model.py` | MLP with Fourier time embedding (demo) |
| `GenericEnergyEvaluator` | `components/generic_evaluator.py` | **NEW** — energy W2 eval for non-particle systems |
| `RotatedGMMEvaluator` | `components/generic_evaluator.py` | **NEW** — adds mode coverage to GenericEnergyEvaluator |
| `RotatedGMMEnergy` | `energies/rotated_gmm_energy.py` | **NEW** — synthetic CV-unknown benchmark |
| `MullerBrownEnergy` | `energies/muller_brown_energy.py` | **NEW** — 2D visualization benchmark |
| `BayesianLogRegEnergy` | `energies/bayesian_logreg_energy.py` | **NEW** — non-molecular posterior sampling |

### Benchmarks

| Benchmark | Dim | Particles | Energy | Expected KSD Impact |
|-----------|-----|-----------|--------|---------------------|
| DW4 | 8 | 4 × 2D | Double well (multimodal) | **High** — clear mode collapse target |
| LJ13 | 39 | 13 × 3D | Lennard-Jones | **Medium** — Stein kernel still works at 39D |
| LJ55 | 165 | 55 × 3D | Lennard-Jones | **Low** — RBF kernel degrades at 165D |

### Hyperparameters (KSD-specific)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ksd_lambda` | 1.0 | KSD penalty weight. Ablate over {0.1, 0.5, 1.0, 5.0, 10.0} |
| `ksd_bandwidth` | null (median heuristic) | RBF bandwidth ℓ. null = auto from data |
| `ksd_max_particles` | 2048 | Subsample if N exceeds this |
| `ksd_efficient_threshold` | 1024 | Use chunked computation above this |

## Conventions
- Conda env: **`Sampling_env`** (NOT `SML_env` — this project needs bgflow + einops)
- Run scripts: `conda run -n Sampling_env python -u <script>.py`
- Hydra outputs: `outputs/EXPERIMENT_NAME/SEED/` (config.yaml + checkpoints/)
- Reference data: downloaded via `scripts/download.sh`
- Config override: `python train.py experiment=dw4_ksd_asbs ksd_lambda=0.5 seed=0`

## Math Reference (see `math_specs.md` for full proofs)
- **Stein kernel**: kₚ(x,x') = K·[s^Ts' + s^Tδ/ℓ² - s'^Tδ/ℓ² + d/ℓ² - r²/ℓ⁴]
- **Detached gradient**: ∇ₓkₚ ≈ K·[-δ/ℓ²·Γ + (s-s')/ℓ² - 2δ/ℓ⁴] (no Hessian)
- **Self-annealing**: KSD term vanishes at convergence (ρ=p → KSD²=0)
- **SVGD connection**: KSD gradient = SVGD update direction

## Gotchas
1. **DO NOT modify existing files** (`matcher.py`, `train.py`, `evaluator.py`, etc.). New code inherits/extends only.
2. **Device consistency** — all tensors in `stein_kernel.py` must stay on same device (CUDA).
3. **COM-free coordinates** — DW4/LJ samples are already center-of-mass free; no special handling needed for Stein kernel.
4. **Gradient detaching** — `_apply_ksd_correction` runs under `@torch.no_grad()`. The KSD correction modifies the adjoint *target* (fixed regression target), NOT the loss. This is by design.
5. **Memory** — N²×D×4 bytes for pairwise tensors. N=512,D=8: 8MB (fine). N=512,D=165: 170MB (ok). N>1024 with high D: use chunked version.
6. **λ tuning** — if `ksd_grad_norm >> adjoint_norm`, λ is too large. Start at 1.0, reduce to 0.1 if unstable.
7. **Hydra `_target_`** — new matcher configs must point to `adjoint_samplers.components.ksd_matcher.KSDAdjointVEMatcher`.
