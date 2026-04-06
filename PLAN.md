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

### 1.9 Validation

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

- [ ] **LJ38 baseline** (~8 hrs) — `experiment=lj38_asbs seed=0`
  - 38 particles × 3D = 114D. Between LJ13 (39D) and LJ55 (165D).
  - Double-funnel energy landscape — ideal mode-collapse test case.
  - **Single seed=0 for focused 3-run comparison** (Baseline / RBF / IMQ+β).
  - Updated: sigma_max=2, batch_size=256, adj_epochs_per_stage=250.
  - [ ] seed 0

- [ ] **LJ55 baseline** (3 seeds × ~12 hrs) — `experiment=lj55_asbs seed=0,1,2`
  - [ ] seed 0
  - [ ] seed 1
  - [ ] seed 2

### 2.2 Visualization Benchmark

- [ ] **Müller-Brown baseline** (3 seeds × ~10 min) — `experiment=muller_asbs seed=0,1,2` ⚠️ NaN errors — retry later

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

### 3.3 LJ38 Focused 3-Run Comparison (seed=0, ~24 GPU-hrs total)

LJ38 uses the **double-funnel** energy landscape (icosahedral vs FCC basins) as a natural mode-collapse test.
At d=114, this sits in the **IMQ kernel sweet spot** (d=30–50 crossover confirmed in RotGMM §5.7).

**New: Temperature-Scaled KSD (β parameter).**
Replace the score in the Stein kernel with a smoothed score s(x) = -β∇E(x), the score of p(x) at effective temperature 1/β. At β=0.1, the icosahedral-FCC barrier (~1 energy unit) shrinks to ~0.1 in the Stein kernel's view. The KSD gradient can "see across" to the FCC funnel. The SDE dynamics and per-particle adjoint still use the true score — only the inter-particle correction uses the smoothed score.

| Run | Config | Kernel | λ | σ_max | β | Purpose |
|-----|--------|--------|---|-------|---|---------|
| 1 | `lj38_asbs` | — | 0 | 2 | — | Control (baseline) |
| 2 | `lj38_ksd_asbs` | RBF | 1.0 | 2 | 1.0 | Does KSD help at d=114? |
| 3 | `lj38_imq_asbs` | IMQ | 1.0 | 5 | 0.1 | Best shot at both funnels |

- [ ] **Run 1: Baseline** — `experiment=lj38_asbs seed=0`
- [ ] **Run 2: KSD-ASBS (RBF)** — `experiment=lj38_ksd_asbs seed=0`
- [ ] **Run 3: KSD-ASBS (IMQ, aggressive)** — `experiment=lj38_imq_asbs seed=0`

#### Expected Outcomes

| Run | Energy histogram | Metrics vs baseline |
|-----|-----------------|---------------------|
| 1. Baseline | Single peak, -170 to -173 (icosahedral only) | — |
| 2. RBF | Single peak, tighter, mean closer to -173.93 | KSD² ↓, dist_W2 ↓ (modest) |
| 3. IMQ+β=0.1 | **Bimodal if successful**: -173.93 (ico) + -173.25 (FCC) | Large KSD² ↓ if both funnels found |

#### Evaluation Protocol

1. Check energy histogram for bimodality.
2. If bimodal: count samples per funnel (E < -172, classify by structure or Q6 order parameter).
3. If unimodal: report intra-funnel diversity improvement.
4. If no improvement: confirms kernel scaling limit at d=114.

### 3.4 LJ55 KSD (best λ × 3 seeds = 3 runs, ~36 GPU-hrs)

Only if LJ38 shows improvement.

- [ ] **λ=best, seed=0,1,2**

### 3.5 Müller-Brown KSD (3 seeds, ~30 min)

- [ ] **λ=1.0, seed=0,1,2** ⚠️ NaN errors — retry later

### 3.6 Bayesian LogReg KSD (2 datasets × 3 seeds = 6 runs, ~2 hrs)

- [ ] **Australian, λ=1.0, seed=0,1,2**
- [ ] **German, λ=1.0, seed=0,1,2**

### 3.7 Ablation: Bandwidth ℓ (DW4, λ=1.0, seed=0)

The RBF bandwidth ℓ is currently set by the **median heuristic** (ℓ = median pairwise distance). In high dimensions, pairwise distances concentrate around σ√D, making the kernel nearly flat. This ablation tests whether a better ℓ improves results.

- [ ] **ℓ = 0.1 × median** — sharper kernel, more local repulsion
- [ ] **ℓ = 0.5 × median** — moderate sharpening
- [ ] **ℓ = 1.0 × median** — current default (baseline for ablation)
- [ ] **ℓ = 2.0 × median** — smoother kernel
- [ ] **ℓ = 5.0 × median** — very smooth, long-range repulsion

**Motivation:** The median heuristic is a one-size-fits-all default. If a non-median ℓ improves DW4 results, that suggests per-benchmark tuning of ℓ (or a better adaptive rule) is worthwhile.

**Implementation:** Add `ksd_bandwidth_scale` config parameter to `ksd_matcher.py`. When set, multiply the median bandwidth by this factor. Default = 1.0 (no change).

### 3.8 Ablation: IMQ Kernel vs RBF (RotGMM d=10, d=30, d=50)

The **Inverse Multi-Quadric (IMQ)** kernel k(x,y) = (c² + ‖x-y‖²)^{-1/2} has heavier tails than RBF, meaning it doesn't vanish as fast when points are far apart. This is theoretically better in high-D where pairwise distances concentrate (Gorham & Mackey, 2017).

- [ ] **Implement IMQ kernel** in `stein_kernel.py`: Stein kernel + gradient (closed-form, similar structure to RBF)
- [ ] **Add `ksd_kernel` config option** to `ksd_matcher.py`: `"rbf"` (default) or `"imq"`
- [ ] **RotGMM d=10** — IMQ KSD-ASBS, λ=1.0, seed=0
- [ ] **RotGMM d=30** — IMQ KSD-ASBS, λ=1.0, seed=0
- [ ] **RotGMM d=50** — IMQ KSD-ASBS, λ=1.0, seed=0
- [ ] **Compare:** mode coverage + energy_W2 for RBF vs IMQ at each dimension

**Hypothesis:** IMQ should maintain mode-resolving power at d=30–50 where RBF degrades. If confirmed, IMQ becomes the recommended kernel for high-D applications.

**IMQ Stein kernel math:**
```
k(x,x') = (c² + ‖x-x'‖²)^{-1/2}
∇ₓ k = (c² + r²)^{-3/2} · (x'-x)
∇²ₓ k = (c² + r²)^{-3/2} · [D - 3r²/(c² + r²)]
Stein kernel: kₚ(x,x') = k · [s^T s' + s^T ∇ₓ' k / k + ∇ₓ k^T s' / k + ∇²ₓₓ' k / k]
```

### 3.9 Ablation: Chunking (DW4, λ=1.0, seed=0)

- [ ] Full computation (ksd_efficient_threshold=99999)
- [ ] Chunked (ksd_efficient_threshold=0)
- Verify: identical loss curves, measure wall-clock overhead

### 3.10 Ablation: Batch Size (DW4, λ=1.0, seed=0)

- [ ] resample_batch_size ∈ {64, 128, 256, 512, 1024}

---

## Phase 4: Synthetic CV-Unknown Experiments

### 4.1 Rotated GMM (4 dims × 2 methods × 3 seeds = 24 runs, ~8 GPU-hrs)

- [ ] **d=10** — baseline + KSD, seed=0,1,2
- [ ] **d=30** — baseline + KSD, seed=0,1,2
- [ ] **d=50** — baseline + KSD, seed=0,1,2
- [ ] **d=100** — baseline + KSD, seed=0,1,2

Key metric: **mode coverage fraction** (not W2). KSD-ASBS should find more modes.

### 4.2 WT-ASBS vs KSD-ASBS on RotGMM (direct comparison)

Port WT-ASBS's well-tempered metadynamics bias to our RotGMM setup. Give WT-ASBS the **correct CVs** (or a subset), while KSD-ASBS gets **no CVs**. This is the fairest head-to-head comparison — same problem, same architecture, different bias mechanisms.

- [ ] Implement WT metadynamics bias for RotGMM in our codebase
- [ ] **d=10** — WT-ASBS (with correct CVs) vs KSD-ASBS (CV-free), seed=0
- [ ] **d=30** — same comparison
- [ ] Argument: *"KSD-ASBS matches or beats WT-ASBS without requiring CV knowledge"*

---

## Phase 5: Comprehensive Evaluation

### 5.1 Create `evaluate_all.py`

- [x] Master eval script — `evaluation/evaluate_all.py` ✅ (file exists)
- [ ] Metrics computed:
  - `energy_w2`, `dist_w2`, `eq_w2` (molecular benchmarks)
  - `ksd_squared`, `mean_energy`, `std_energy`, `min_energy`, `max_energy`
  - `n_modes_covered`, `coverage_fraction` (RotGMM only)
- [ ] Saves per-group JSON files to `results/`

### 5.2 LJ38 Data-Free Evaluation Metrics

Since ground-truth reference samples are unavailable for LJ38, we define **four data-free metrics** that exploit only the known Lennard-Jones potential $U(x)$ and the model's own log-probability $\log q_\theta(x)$.

#### 5.2.1 Steinhardt Bond-Orientational Order Parameters ($Q_4$ and $Q_6$)

**Purpose:** Classify generated 3D coordinates into the two dominant LJ38 funnels without visual inspection.
- FCC Truncated Octahedron (Global Min): High $Q_4$, specific $Q_6$.
- Mackay Icosahedron (Secondary Min): High $Q_6$, near-zero $Q_4$.

**Implementation:**
1. Take the generated sample tensor of shape `(batch_size, 38, 3)`.
2. Do not write spherical harmonics from scratch. Use an established physics library like **freud** (Python) or **pyscal** to compute the per-atom or global $Q_4$ and $Q_6$ for each sample in the batch.
3. Output: Return a `(batch_size, 2)` tensor containing the $(Q_4, Q_6)$ pairs.
4. Visualization: Implement a function to plot a 2D histogram (hexbin or scatter with density) of these pairs to visually confirm the two distinct funnel clusters.

#### 5.2.2 Free Energy Difference ($\Delta F$)

**Purpose:** Measure if the model captures the correct thermodynamic probability weights between the two funnels at a specific temperature.

**Implementation:**
1. Define boundaries in the $(Q_4, Q_6)$ space to classify a sample as belonging to State A (FCC) or State B (Icosahedral).
2. Count the number of samples in each state ($N_A$ and $N_B$).
3. Calculate:
   $$\Delta F_{A \to B} = -k_B T \ln \left( \frac{N_B}{N_A} \right)$$
   *(Set $k_B = 1$ and use the simulation temperature $T$ if working in reduced LJ units.)*
4. Output: Return the scalar $\Delta F$ value.

#### 5.2.3 Unnormalized Reverse KL Divergence

**Purpose:** Evaluate model convergence to the true Boltzmann distribution without ground-truth samples.

**Implementation:**
1. Inputs: model's log-probability $\log q_\theta(x)$ for each generated sample, and the true LJ potential $U(x)$.
2. Calculate:
   $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[ \log q_\theta(x_i) + \beta U(x_i) \right]$$
   where $\beta = 1 / (k_B T)$.
3. Constraint: Ensure $U(x)$ is implemented efficiently in PyTorch to process the `(batch_size, 38, 3)` tensor in parallel without CPU bottlenecks.
4. Output: Return the scalar $\mathcal{L}$. Lower values indicate a better fit.

#### 5.2.4 Effective Sample Size (ESS)

**Purpose:** Measure mode collapse and sampling efficiency. ESS close to $N$ = perfect sampling; ESS near $1$ = severe mode collapse (e.g., stuck in one funnel).

**Implementation:**
1. Compute unnormalized log-importance weights for each sample:
   $$\log w_i = -\beta U(x_i) - \log q_\theta(x_i)$$
2. **Crucial:** Use the `torch.logsumexp` trick to prevent underflow/overflow:
   $$\log(\text{ESS}) = 2 \cdot \text{logsumexp}(\log w_i) - \text{logsumexp}(2 \cdot \log w_i)$$
   $$\text{ESS} = \exp(\log(\text{ESS}))$$
3. Output: Return the ESS value and the ESS ratio ($\text{ESS} / \text{batch\_size}$).

#### General Directives

- All code must be well-structured; all inline comments written entirely in English.
- Ensure tensor operations are batched for PyTorch.
- These metrics replace the W2-based metrics (which require reference samples) for LJ38 evaluation.

### 5.3 Chunking Timing Test

- [ ] Compare full vs chunked Stein kernel gradient: N=512, d∈{8, 39, 165}
- [ ] Verify max absolute difference ≈ 0 (fp precision)

### 5.4 Run Evaluation

- [ ] `python evaluation/evaluate_all.py --outputs_root outputs --results_dir results --n_samples 2000 --n_eval_seeds 5`

---

## Phase 6: Results Report

### 6.1 Create `generate_results.py`

- [x] `evaluation/generate_results.py` — reads results, generates RESULTS.md ✅ (file exists)
- [ ] Tables: per-benchmark comparison (baseline vs KSD), λ ablation, mode coverage
- [ ] Chunking timing table
- [ ] **Visualization ideas for modality comparison:**
  - **RotGMM mode occupation bar chart** — X=mode index, Y=fraction of samples per mode. Vanilla ASBS has missing bars (mode collapse), KSD-ASBS fills them all. If WT-ASBS comparison done, overlay its bars too.
  - **RotGMM 2D scatter** (d=10, project onto top 2 PCs) — color by assigned mode. Vanilla misses clusters, KSD covers all.
  - **LJ13/LJ38 energy histogram** — Vanilla clusters in one basin, KSD shows bimodal distribution matching reference. Especially powerful for LJ38 double funnel.
  - **LJ13/LJ38 pairwise distance distribution** — shows structural diversity of generated samples.
  - **Müller-Brown 2D density contour** — overlay generated sample density on energy landscape. Vanilla misses a well, KSD covers all three. (Prettiest figure — if NaN fixed.)
  - **KSD vs training epoch** — convergence plot showing KSD-ASBS reaches lower discrepancy faster than vanilla.
  - **Free energy surface comparison** — -kT ln(p) along principal components, comparing vanilla vs KSD vs reference.

### 6.2 Generate Final Report

- [ ] `python evaluation/generate_results.py --results_dir results --output evaluation/RESULTS.md`
- [ ] Review, interpret, write conclusions

---

## Priority Order (if compute-limited)

| Priority | Experiment | Est. Time | Why |
|----------|-----------|-----------|-----|
| 1 | DW4 baseline + KSD | ~3 hrs | ✅ **DONE** — proves concept |
| 2 | Müller-Brown baseline + KSD | ~30 min | ✅ **DONE** — 2D visualization |
| 3 | RotGMM d=10,30 | ~2 hrs | ✅ **DONE** — proves CV-unknown advantage |
| 4 | RotGMM d=50,100 | ~2 hrs | ✅ **DONE (d=50)**, d=100 training — dimension scaling |
| 5 | **IMQ kernel on RotGMM d=10,30,50** | ~3 hrs | **High-D fix — could recover mode coverage at d=30–50** |
| 6 | **DW4 ℓ ablation** | ~5 hrs | Bandwidth sensitivity — informs optimal kernel tuning |
| 7 | DW4 λ ablation | ~5 hrs | Finds optimal hyperparameter |
| 8 | Bayesian LogReg | ~2 hrs | Proves generality beyond molecular |
| 9 | LJ13 baseline + KSD | ~24 hrs | Medium-scale molecular |
| 10 | **LJ38 3-run comparison** (baseline / RBF / IMQ+β) | ~24 hrs | **114D molecular — double-funnel, temperature-scaled KSD** |
| 11 | LJ55 baseline + KSD | ~72 hrs | Only if LJ38 works |

**Total estimated GPU time: ~250–300 hours (parallelizable across seeds).**

---

## Gotchas & Notes

1. **DO NOT modify existing files** (`matcher.py`, `train.py`, `evaluator.py`, etc.). All new code inherits/extends.
2. **Non-graph vs graph**: RotGMM/Müller/BLogReg use `VESDE` + `FourierMLP` + `gauss` source. Molecular systems use `GraphVESDE` + `EGNN` + `harmonic` source.
3. **RotGMM has n_particles=1**: sidesteps graph machinery.
4. **λ tuning**: if `ksd_grad_norm >> adjoint_norm`, λ is too large. Start at 1.0, reduce to 0.1 if unstable.
6. **Memory**: Stein kernel gradient is O(N²d). At N=512, d=165: ~170MB — fine. Use chunked version above N=1024.
7. **Seeding**: training seeds (0,1,2) ≠ evaluation seeds (eval_seed in evaluate_all.py).
8. **Hydra outputs**: `outputs/{exp_name}/` with `checkpoints/` and `.hydra/config.yaml`.
