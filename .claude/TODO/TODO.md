# TODO — Stein_ASBS

## Implementation Tasks (in order)

### Phase 1: Core Modules
- [ ] **Task 1** — Create `adjoint_samplers/components/stein_kernel.py`
  - `median_bandwidth()`, `compute_ksd_squared()`, `compute_stein_kernel_gradient()`, `compute_stein_kernel_gradient_efficient()`
  - Run unit test: `python -m adjoint_samplers.components.stein_kernel` → "All Stein kernel tests passed!"
- [ ] **Task 2** — Create `adjoint_samplers/components/ksd_matcher.py`
  - `KSDAdjointVEMatcher(AdjointVEMatcher)` — overrides `populate_buffer` to add KSD correction
  - `KSDAdjointVPMatcher(AdjointVPMatcher)` — same for VP-SDE
  - Key method: `_apply_ksd_correction(x1, adjoint1)` → computes scores, bandwidth, Stein gradient, applies correction

### Phase 2: Configs
- [ ] **Task 3.1** — Create `configs/matcher/ksd_adjoint_ve.yaml` (points to `KSDAdjointVEMatcher`)
- [ ] **Task 3.2** — Create `configs/experiment/dw4_ksd_asbs.yaml` (copy dw4_asbs, swap matcher, add `ksd_lambda`)
- [ ] **Task 3.3** — Create `configs/experiment/lj13_ksd_asbs.yaml`
- [ ] **Task 3.4** — Create `configs/experiment/lj55_ksd_asbs.yaml`

### Phase 3: Training & Validation
- [ ] **Task 4** — (Optional) Add KSD logging to `train.py` (`_last_ksd`, `_last_ksd_grad_norm`)
- [ ] Train DW4 baseline: `python train.py experiment=dw4_asbs seed=0,1,2 -m`
- [ ] Train DW4 KSD-ASBS: `python train.py experiment=dw4_ksd_asbs seed=0,1,2 -m ksd_lambda=1.0`
- [ ] Verify KSD-ASBS trains without crashing; compare loss curves to baseline
- [ ] λ ablation on DW4: {0.1, 0.5, 1.0, 5.0, 10.0}

### Phase 4: Evaluation & Reporting
- [ ] **Task 5** — Create `evaluate_comparison.py` (head-to-head baseline vs KSD-ASBS)
- [ ] **Task 6** — Create `generate_results.py` (auto-generate RESULTS.md with tables/charts)
- [ ] Run DW4 comparison
- [ ] Run LJ13 comparison
- [ ] Run LJ55 comparison (only if DW4/LJ13 show improvement)

### Phase 5: Scale Up
- [ ] Train LJ13 KSD-ASBS (3 seeds)
- [ ] Train LJ55 KSD-ASBS (only if DW4/LJ13 positive)

## Success Criteria
- **Primary**: Lower KSD², lower energy W₂, lower eq_W₂ vs baseline
- **Secondary**: Mean energy not degraded
- **Self-annealing check**: KSD correction should shrink during training as ρ→p

## Known Issues
- **LJ55 (165D)**: RBF kernel likely degrades — pairwise distances concentrate. May need deep/graph kernel for high-D (future work).
- **Hessian-free approximation**: Drops H(x) terms from Stein kernel gradient. Justified for well-separated particles but may be too crude in some regimes.
- **O(N²d) overhead**: Negligible for DW4 (8D) and LJ13 (39D), comparable to SDE cost for LJ55 (165D). Use subsampling (`ksd_max_particles`) if needed.

## Negative Result Interpretation
- KSD-ASBS ≈ baseline → ASBS already covers modes well (KSD vanishes at convergence)
- KSD-ASBS worse → detached Hessian approx too crude, or λ needs tuning
- DW4 improved, LJ55 not → kernel scaling limitation (publishable observation)
