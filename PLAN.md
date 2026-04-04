# Execution Plan: KSD-Augmented ASBS — Full Experimental Pipeline

**Generated:** 2026-04-04 KST
**Source:** `.claude/TODO/implementation_guide.md`

---

## Phase 1: Infrastructure — Code & Configs

All new source files, energy functions, evaluators, and Hydra configs needed before any training.

### 1.1 Core KSD Modules

- [x] **Create `adjoint_samplers/components/stein_kernel.py`** ✅
- [x] **Create `adjoint_samplers/components/ksd_matcher.py`** ✅

### 1.2 New Energy Functions

- [x] **Create `adjoint_samplers/energies/rotated_gmm_energy.py`** — `RotatedGMMEnergy` ✅
- [x] **Create `adjoint_samplers/energies/muller_brown_energy.py`** — `MullerBrownEnergy` ✅
- [x] **Create `adjoint_samplers/energies/bayesian_logreg_energy.py`** — `BayesianLogRegEnergy` ✅
  - HMC reference generation is now **lazy** (runs on first `get_ref_samples()` call, not at init)

### 1.3 New Evaluators

- [x] **Create `RotatedGMMEvaluator`** in `generic_evaluator.py` ✅

### 1.4 Hydra Configs — Matchers

- [x] **`configs/matcher/ksd_adjoint_ve.yaml`** ✅

### 1.5 Hydra Configs — Problems

- [x] `configs/problem/dw4.yaml`, `lj13.yaml`, `lj55.yaml` ✅
- [x] `configs/problem/lj38.yaml` ✅
- [x] `configs/problem/muller.yaml` ✅
- [x] `configs/problem/blogreg_au.yaml`, `blogreg_ge.yaml` ✅
- [x] `configs/problem/rotgmm10.yaml`, `rotgmm30.yaml`, `rotgmm50.yaml`, `rotgmm100.yaml` ✅

### 1.6 Hydra Configs — Experiments (Baseline)

- [x] `configs/experiment/dw4_asbs.yaml`, `lj13_asbs.yaml`, `lj55_asbs.yaml` ✅
- [x] `configs/experiment/lj38_asbs.yaml` ✅
- [x] `configs/experiment/muller_asbs.yaml` ✅
- [x] `configs/experiment/blogreg_au_asbs.yaml`, `blogreg_ge_asbs.yaml` ✅
- [x] `configs/experiment/rotgmm10_asbs.yaml`, `rotgmm30_asbs.yaml`, `rotgmm50_asbs.yaml`, `rotgmm100_asbs.yaml` ✅

### 1.7 Hydra Configs — Experiments (KSD-ASBS)

- [x] `configs/experiment/dw4_ksd_asbs.yaml` ✅
- [x] `configs/experiment/lj13_ksd_asbs.yaml` ✅
- [x] `configs/experiment/lj38_ksd_asbs.yaml` ✅
- [x] `configs/experiment/lj55_ksd_asbs.yaml` ✅
- [x] `configs/experiment/muller_ksd_asbs.yaml` ✅
- [x] `configs/experiment/blogreg_au_ksd_asbs.yaml`, `blogreg_ge_ksd_asbs.yaml` ✅
- [x] `configs/experiment/rotgmm10_ksd_asbs.yaml`, `rotgmm30_ksd_asbs.yaml`, `rotgmm50_ksd_asbs.yaml`, `rotgmm100_ksd_asbs.yaml` ✅

### 1.8 Misc Configs

- [x] `configs/sde/ve.yaml` ✅
- [x] `configs/term_cost/score_term_cost.yaml` ✅
- [x] `configs/source/gauss.yaml` ✅

### 1.9 LJ38 Reference Data

- [ ] **Generate `data/test_split_LJ38-1000.npy`** ⚠️ STILL MISSING — no public download exists
  - Must self-generate via parallel-tempering MCMC or long MD at target temperature
  - Need ~500 samples from each funnel (icosahedral: E≈-173.93, FCC: E≈-173.25)
  - **Deferred** — do this before LJ38 evaluation (Phase 5), not blocking other experiments

### 1.10 Validation

- [x] Verify RotGMM energy: forward pass ✅
- [x] Verify Müller-Brown energy: forward pass ✅
- [x] Verify BayesianLogReg energy: forward pass ✅ (lazy HMC fix applied)
- [x] Run `python -m adjoint_samplers.components.stein_kernel` → "All Stein kernel tests passed!" ✅
- [ ] Verify Hydra config resolution: `python train.py experiment=dw4_ksd_asbs --cfg job` (dry run)

---

## Phase 2: Baseline ASBS Training

Train all baselines with 3 training seeds each. **~50 GPU-hours total.**

### 2.1 Molecular Benchmarks

- [x] **DW4 baseline** (3 seeds × ~1 hr) — `experiment=dw4_asbs seed=0,1,2`
  - ✅ seed 0 done (existing `baselines/dw4_asbs/`)
  - [ ] seed 1
  - [ ] seed 2

- [ ] **LJ13 baseline** (3 seeds × ~4 hrs) — `experiment=lj13_asbs seed=0,1,2`
  - Existing `baselines/lj13_asbs/` has 1 checkpoint
  - [ ] seed 0 (may already exist)
  - [ ] seed 1
  - [ ] seed 2

- [ ] **LJ38 baseline** (3 seeds × ~8 hrs) — `experiment=lj38_asbs seed=0,1,2`
  - Requires LJ38 reference data (Phase 1.9)
  - [ ] seed 0
  - [ ] seed 1
  - [ ] seed 2

- [ ] **LJ55 baseline** (3 seeds × ~12 hrs) — `experiment=lj55_asbs seed=0,1,2`
  - [ ] seed 0
  - [ ] seed 1
  - [ ] seed 2

### 2.2 Visualization Benchmark

- [ ] **Müller-Brown baseline** (3 seeds × ~10 min) — `experiment=muller_asbs seed=0,1,2` 🔄 RUNNING seed=0

### 2.3 Non-Molecular Benchmarks

- [ ] **Bayesian LogReg — Australian** (3 seeds × ~20 min) — `experiment=blogreg_au_asbs seed=0,1,2`
- [ ] **Bayesian LogReg — German** (3 seeds × ~20 min) — `experiment=blogreg_ge_asbs seed=0,1,2`

### 2.4 Synthetic CV-Unknown

- [ ] **RotGMM d=10** (3 seeds × ~20 min) — `experiment=rotgmm10_asbs seed=0,1,2`
- [ ] **RotGMM d=30** (3 seeds × ~20 min) — `experiment=rotgmm30_asbs seed=0,1,2`
- [ ] **RotGMM d=50** (3 seeds × ~20 min) — `experiment=rotgmm50_asbs seed=0,1,2`
- [ ] **RotGMM d=100** (3 seeds × ~20 min) — `experiment=rotgmm100_asbs seed=0,1,2`

---

## Phase 3: KSD-ASBS Training + λ Ablation

### 3.1 DW4 λ Ablation (5λ × 3 seeds = 15 runs, ~15 GPU-hrs)

- [x] **λ=1.0** — ✅ Done (existing `results/local/2026.04.04/064017/`)
- [ ] **λ=0.1, seed=0,1,2**
- [ ] **λ=0.5, seed=0,1,2**
- [ ] **λ=1.0, seed=1,2** (seed 0 done)
- [ ] **λ=5.0, seed=0,1,2**
- [ ] **λ=10.0, seed=0,1,2**

### 3.2 LJ13 KSD (3λ × 3 seeds = 9 runs, ~36 GPU-hrs)

Use best λ from DW4 ± neighbors.

- [ ] **λ=0.5, seed=0,1,2**
- [ ] **λ=1.0, seed=0,1,2**
- [ ] **λ=5.0, seed=0,1,2**

### 3.3 LJ38 KSD (3λ × 3 seeds = 9 runs, ~72 GPU-hrs)

The headline experiment — double-funnel landscape.

- [ ] **λ=0.5, seed=0,1,2**
- [ ] **λ=1.0, seed=0,1,2**
- [ ] **λ=5.0, seed=0,1,2**

### 3.4 LJ55 KSD (best λ × 3 seeds = 3 runs, ~36 GPU-hrs)

Only if DW4/LJ13/LJ38 show improvement.

- [ ] **λ=best, seed=0,1,2**

### 3.5 Müller-Brown KSD (3 seeds, ~30 min)

- [ ] **λ=1.0, seed=0,1,2**

### 3.6 Bayesian LogReg KSD (2 datasets × 3 seeds = 6 runs, ~2 hrs)

- [ ] **Australian, λ=1.0, seed=0,1,2**
- [ ] **German, λ=1.0, seed=0,1,2**

### 3.7 Ablation: Chunking (DW4, λ=1.0, seed=0)

- [ ] Full computation (ksd_efficient_threshold=99999)
- [ ] Chunked (ksd_efficient_threshold=0)
- Verify: identical loss curves, measure wall-clock overhead

### 3.8 Ablation: Batch Size (DW4, λ=1.0, seed=0)

- [ ] resample_batch_size ∈ {64, 128, 256, 512, 1024}

---

## Phase 4: Synthetic CV-Unknown Experiments

### 4.1 Rotated GMM (4 dims × 2 methods × 3 seeds = 24 runs, ~8 GPU-hrs)

- [ ] **d=10** — baseline + KSD, seed=0,1,2
- [ ] **d=30** — baseline + KSD, seed=0,1,2
- [ ] **d=50** — baseline + KSD, seed=0,1,2
- [ ] **d=100** — baseline + KSD, seed=0,1,2

Key metric: **mode coverage fraction** (not W2). KSD-ASBS should find more modes.

---

## Phase 5: Comprehensive Evaluation

### 5.1 Create `evaluate_all.py`

- [x] Master eval script — `evaluation/evaluate_all.py` ✅ (file exists)
- [ ] Metrics computed:
  - `energy_w2`, `dist_w2`, `eq_w2` (molecular benchmarks)
  - `ksd_squared`, `mean_energy`, `std_energy`, `min_energy`, `max_energy`
  - `n_modes_covered`, `coverage_fraction` (RotGMM only)
- [ ] Saves per-group JSON files to `results/`

### 5.2 Chunking Timing Test

- [ ] Compare full vs chunked Stein kernel gradient: N=512, d∈{8, 39, 114, 165}
- [ ] Verify max absolute difference ≈ 0 (fp precision)

### 5.3 Run Evaluation

- [ ] `python evaluation/evaluate_all.py --outputs_root outputs --results_dir results --n_samples 2000 --n_eval_seeds 5`

### 5.4 LJ38 Funnel Classification

- [ ] For each generated LJ38 sample: compute energy, classify as icosahedral or FCC funnel
- [ ] Count: how many samples in each funnel?
- [ ] **Headline result:** if baseline finds 1 funnel, KSD-ASBS finds both → mode collapse fixed

---

## Phase 6: Results Report

### 6.1 Create `generate_results.py`

- [x] `evaluation/generate_results.py` — reads results, generates RESULTS.md ✅ (file exists)
- [ ] Tables: per-benchmark comparison (baseline vs KSD), λ ablation, mode coverage
- [ ] Figures: energy histograms, λ ablation curves, mode coverage bar chart, Müller-Brown landscape
- [ ] Chunking timing table

### 6.2 Generate Final Report

- [ ] `python evaluation/generate_results.py --results_dir results --output evaluation/RESULTS.md`
- [ ] Review, interpret, write conclusions

---

## Priority Order (if compute-limited)

| Priority | Experiment | Est. Time | Why |
|----------|-----------|-----------|-----|
| 1 | DW4 baseline + KSD | ~3 hrs | ✅ **DONE** — proves concept |
| 2 | Müller-Brown baseline + KSD | ~30 min | Best visualization figure |
| 3 | RotGMM d=10,30 | ~2 hrs | Proves CV-unknown advantage |
| 4 | DW4 λ ablation | ~5 hrs | Finds optimal hyperparameter |
| 5 | LJ38 baseline + KSD | ~24 hrs | Headline real-system result (double funnel) |
| 6 | Bayesian LogReg | ~2 hrs | Proves generality beyond molecular |
| 7 | LJ13 baseline + KSD | ~24 hrs | Medium-scale molecular |
| 8 | RotGMM d=50,100 | ~2 hrs | Dimension scaling |
| 9 | LJ55 baseline + KSD | ~72 hrs | Only if LJ38 works |

**Total estimated GPU time: ~250–300 hours (parallelizable across seeds).**

---

## Gotchas & Notes

1. **DO NOT modify existing files** (`matcher.py`, `train.py`, `evaluator.py`, etc.). All new code inherits/extends.
2. **Non-graph vs graph**: RotGMM/Müller/BLogReg use `VESDE` + `FourierMLP` + `gauss` source. Molecular systems use `GraphVESDE` + `EGNN` + `harmonic` source.
3. **RotGMM has n_particles=1**: sidesteps graph machinery.
4. **LJ38 reference data** must be generated or downloaded before training can be evaluated.
5. **λ tuning**: if `ksd_grad_norm >> adjoint_norm`, λ is too large. Start at 1.0, reduce to 0.1 if unstable.
6. **Memory**: Stein kernel gradient is O(N²d). At N=512, d=165: ~170MB — fine. Use chunked version above N=1024.
7. **Seeding**: training seeds (0,1,2) ≠ evaluation seeds (eval_seed in evaluate_all.py).
8. **Hydra outputs**: `outputs/{exp_name}/` with `checkpoints/` and `.hydra/config.yaml`.
