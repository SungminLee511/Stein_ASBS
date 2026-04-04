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
│   │   ├── SKILL.md              # This file — project conventions & architecture
│   │   └── math_specs.md         # Full math derivation (Stein kernel, KSD, adjoint eqs)
│   └── TODO/
│       ├── TODO.md               # Open tasks & known issues
│       └── implementation_guide.md  # Step-by-step build plan with full code
├── train.py                      # Entry point — Hydra-based training script
├── adjoint_samplers/             # Main package
│   ├── train_loop.py             # Training loop logic
│   ├── components/
│   │   ├── matcher.py            # EXISTING — AdjointVEMatcher, AdjointVPMatcher (DO NOT MODIFY)
│   │   ├── buffer.py             # EXISTING — BatchBuffer for trajectory data
│   │   ├── evaluator.py          # EXISTING — SyntheticEnergyEvaluator
│   │   ├── model.py              # EXISTING — FourierMLP, EGNN architectures
│   │   ├── sde.py                # EXISTING — SDE definitions, sdeint, ControlledSDE
│   │   ├── state_cost.py         # EXISTING — ZeroGradStateCost
│   │   ├── term_cost.py          # EXISTING — terminal cost (score/corrector)
│   │   ├── stein_kernel.py       # NEW (to create) — KSD computation, Stein kernel gradient
│   │   └── ksd_matcher.py        # NEW (to create) — KSDAdjointVE/VPMatcher
│   ├── energies/
│   │   ├── base_energy.py        # Abstract energy interface
│   │   ├── double_well_energy.py # DW4 benchmark (8D, 4 particles × 2D)
│   │   ├── lennard_jones_energy.py # LJ13 (39D) and LJ55 (165D) benchmarks
│   │   └── dist_energy.py        # Distribution-based energy
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
│   │   ├── dw4_as.yaml           # EXISTING — DW4 without corrector
│   │   ├── lj13_asbs.yaml        # EXISTING — LJ13 baseline
│   │   ├── lj13_as.yaml          # EXISTING — LJ13 without corrector
│   │   ├── lj55_asbs.yaml        # EXISTING — LJ55 baseline
│   │   ├── lj55_as.yaml          # EXISTING — LJ55 without corrector
│   │   ├── demo_asbs.yaml        # EXISTING — demo config
│   │   ├── demo_memoryless_soc.yaml
│   │   ├── demo_nonmemoryless_soc.yaml
│   │   ├── dw4_ksd_asbs.yaml     # NEW (to create) — DW4 + KSD
│   │   ├── lj13_ksd_asbs.yaml    # NEW (to create) — LJ13 + KSD
│   │   └── lj55_ksd_asbs.yaml    # NEW (to create) — LJ55 + KSD
│   ├── matcher/
│   │   ├── adjoint_ve.yaml       # EXISTING — VE adjoint matcher
│   │   ├── adjoint_vp.yaml       # EXISTING — VP adjoint matcher
│   │   ├── corrector.yaml        # EXISTING — corrector matcher
│   │   └── ksd_adjoint_ve.yaml   # NEW (to create) — KSD VE matcher
│   ├── sde/                      # ve.yaml, vp.yaml, graph_ve.yaml, etc.
│   ├── problem/                  # dw4.yaml, lj13.yaml, lj55.yaml, demo.yaml
│   ├── source/                   # gauss.yaml, harmonic.yaml, delta.yaml, meanfree.yaml
│   ├── model/                    # fouriermlp.yaml, egnn.yaml
│   ├── state_cost/               # zero.yaml
│   ├── term_cost/                # score_term_cost.yaml, corrector_term_cost.yaml, graph_*
│   └── lancher/                  # Slurm launcher config
├── scripts/
│   ├── dw4.sh                    # DW4 training script
│   ├── lj13.sh                   # LJ13 training script
│   ├── lj55.sh                   # LJ55 training script
│   ├── demo.sh                   # Demo script
│   └── download.sh               # Download reference test samples
├── evaluate_comparison.py        # NEW (to create) — head-to-head eval script
├── generate_results.py           # NEW (to create) — auto-generate RESULTS.md
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
- Conda env: `SML_env`
- Run scripts: `conda run -n SML_env python -u <script>.py`
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
