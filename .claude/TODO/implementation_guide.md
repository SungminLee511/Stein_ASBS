# Comprehensive Experimental Guide: KSD-Augmented ASBS

## Full Implementation, Training, Evaluation, and Results Pipeline

-----

## 0. Overview

This guide covers **every experiment** needed to evaluate KSD-Augmented ASBS comprehensively. The experiments are organized into 6 phases, designed to be executed sequentially. Each phase produces artifacts (checkpoints, metrics, figures) consumed by later phases.

### Experiment Matrix

|Phase|What                    |Benchmarks               |Runs                                          |
|-----|------------------------|-------------------------|----------------------------------------------|
|1    |Infrastructure          |—                        |Setup only                                    |
|2    |Baseline ASBS           |DW4, LJ13, LJ38, LJ55    |3 seeds × 4 benchmarks = 12                   |
|3    |KSD-ASBS + ablation     |DW4, LJ13, LJ38, LJ55    |5 λ × 3 seeds × 4 benchmarks = 60             |
|4    |Synthetic CV-unknown    |RotGMM (d=10,30,50,100)  |3 seeds × 4 dims × 2 methods = 24             |
|4b   |Real CV-unknown         |LJ38, Müller-Brown       |Covered in Phase 2–3 (LJ38), Phase 4b (Müller)|
|4c   |Non-molecular           |Bayesian LogReg (d=15,25)|3 seeds × 2 datasets × 2 methods = 12         |
|5    |Comprehensive evaluation|All above                |All checkpoints                               |
|6    |Results report          |—                        |Generate RESULTS.md                           |

**Total training runs: ~110.** On a single GPU, DW4 trains in ~1 hour, LJ13 in ~4 hours, LJ38 in ~8 hours, LJ55 in ~12 hours, Müller-Brown in ~10 min, Bayesian LogReg in ~20 min, synthetics in ~20 min each. Total estimated GPU time: ~250–300 hours. Parallelizable across seeds.

### Benchmark Summary

|Benchmark       |Dim       |Particles|CVs known?          |Why it matters                                        |
|----------------|----------|---------|--------------------|------------------------------------------------------|
|DW4             |8         |4        |Partially           |Multimodal, fast iteration                            |
|LJ13            |39        |13       |Partially           |Medium-scale particle system                          |
|**LJ38**        |**114**   |**38**   |**No**              |**Double-funnel landscape — icosahedral vs FCC**      |
|LJ55            |165       |55       |No                  |High-dimensional, ~10⁸ local minima                   |
|RotGMM          |10–100    |1        |No (by construction)|Proves CV-free advantage on synthetics                |
|**Müller-Brown**|**2**     |**1**    |**Trivially yes**   |**Visualization — show mode coverage on 2D landscape**|
|**Bayes LogReg**|**15, 25**|**1**    |**No**              |**Non-molecular — shows generality beyond physics**   |

### File Plan

```
adjoint_samplers/                         # EXISTING REPO
├── adjoint_samplers/
│   ├── components/
│   │   ├── stein_kernel.py               # NEW (Phase 1)
│   │   ├── ksd_matcher.py                # NEW (Phase 1)
│   │   └── ...existing...
│   ├── energies/
│   │   ├── rotated_gmm_energy.py         # NEW (Phase 1) — synthetic benchmark
│   │   ├── muller_brown_energy.py        # NEW (Phase 1) — 2D visualization benchmark
│   │   ├── bayesian_logreg_energy.py     # NEW (Phase 1) — non-molecular benchmark
│   │   └── ...existing...
│   └── utils/
│       └── ...existing...
├── configs/
│   ├── experiment/
│   │   ├── dw4_ksd_asbs.yaml             # NEW (Phase 1)
│   │   ├── lj13_ksd_asbs.yaml            # NEW (Phase 1)
│   │   ├── lj38_asbs.yaml                # NEW (Phase 1) — double-funnel benchmark
│   │   ├── lj38_ksd_asbs.yaml            # NEW (Phase 1)
│   │   ├── lj55_ksd_asbs.yaml            # NEW (Phase 1)
│   │   ├── muller_asbs.yaml              # NEW (Phase 1) — 2D visualization
│   │   ├── muller_ksd_asbs.yaml          # NEW (Phase 1)
│   │   ├── blogreg_au_asbs.yaml          # NEW (Phase 1) — Bayesian logistic regression
│   │   ├── blogreg_au_ksd_asbs.yaml      # NEW (Phase 1)
│   │   ├── blogreg_ge_asbs.yaml          # NEW (Phase 1)
│   │   ├── blogreg_ge_ksd_asbs.yaml      # NEW (Phase 1)
│   │   ├── rotgmm10_asbs.yaml            # NEW (Phase 1)
│   │   ├── rotgmm10_ksd_asbs.yaml        # NEW (Phase 1)
│   │   ├── rotgmm30_asbs.yaml            # NEW (Phase 1)
│   │   ├── rotgmm30_ksd_asbs.yaml        # NEW (Phase 1)
│   │   ├── rotgmm50_asbs.yaml            # NEW (Phase 1)
│   │   ├── rotgmm50_ksd_asbs.yaml        # NEW (Phase 1)
│   │   ├── rotgmm100_asbs.yaml           # NEW (Phase 1)
│   │   └── rotgmm100_ksd_asbs.yaml       # NEW (Phase 1)
│   ├── matcher/
│   │   └── ksd_adjoint_ve.yaml           # NEW (Phase 1)
│   ├── problem/
│   │   ├── lj38.yaml                     # NEW (Phase 1)
│   │   ├── muller.yaml                   # NEW (Phase 1)
│   │   ├── blogreg_au.yaml               # NEW (Phase 1)
│   │   ├── blogreg_ge.yaml               # NEW (Phase 1)
│   │   ├── rotgmm10.yaml                 # NEW (Phase 1)
│   │   ├── rotgmm30.yaml                 # NEW (Phase 1)
│   │   ├── rotgmm50.yaml                 # NEW (Phase 1)
│   │   └── rotgmm100.yaml               # NEW (Phase 1)
│   └── ...existing...
├── train.py                              # EXISTING — do not modify
├── scripts/
│   ├── run_phase2_baselines.sh           # NEW (Phase 2)
│   ├── run_phase3_ksd.sh                 # NEW (Phase 3)
│   ├── run_phase4_synthetic.sh           # NEW (Phase 4)
│   ├── run_phase4b_cvunknown.sh          # NEW (Phase 4b)
│   ├── run_phase4c_nonmolecular.sh       # NEW (Phase 4c)
│   └── run_phase5_evaluate.sh            # NEW (Phase 5)
├── evaluate_all.py                       # NEW (Phase 5)
├── generate_results.py                   # NEW (Phase 6)
└── docs/
    ├── STEIN_VARIATIONAL_ASBS.md         # Math spec (already written)
    ├── KSD_ASBS_IMPLEMENTATION.md        # Previous impl guide
    └── RESULTS.md                        # AUTO-GENERATED (Phase 6)
```

-----

## Phase 1: Infrastructure

### 1.1 Stein Kernel Module (`adjoint_samplers/components/stein_kernel.py`)

See `KSD_ASBS_IMPLEMENTATION.md` Task 1 for the complete implementation. This module provides:

- `median_bandwidth(samples)` → scalar
- `compute_stein_kernel_matrix(samples, scores, ell)` → (N, N)
- `compute_ksd_squared(samples, scores, ell)` → scalar
- `compute_stein_kernel_gradient(samples, scores, ell)` → (N, D)
- `compute_stein_kernel_gradient_efficient(samples, scores, ell, chunk_size)` → (N, D)

**Test:** `python -m adjoint_samplers.components.stein_kernel` should print “All tests passed.”

### 1.2 KSD Matcher (`adjoint_samplers/components/ksd_matcher.py`)

See `KSD_ASBS_IMPLEMENTATION.md` Task 2 for the complete implementation. Provides:

- `KSDAdjointVEMatcher` — inherits from `AdjointVEMatcher`, overrides `populate_buffer`
- `KSDAdjointVPMatcher` — same for VP-SDE

### 1.3 Rotated Gaussian Mixture Energy (`adjoint_samplers/energies/rotated_gmm_energy.py`)

This is the synthetic benchmark where CVs are unknown by construction.

```python
"""
adjoint_samplers/energies/rotated_gmm_energy.py

Rotated Gaussian Mixture Model energy function.

After a random rotation R, no axis-aligned projection separates the modes.
This makes CV-based methods (like WT-ASBS) fail — they cannot find the
right projection without oracle knowledge of R.

Our KSD method operates in the full space and doesn't need CVs.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class RotatedGMMEnergy(BaseEnergy):
    """Rotated Gaussian Mixture Model.

    p(x) ∝ Σ_k w_k N(x; R μ_k, σ_k² I)

    where R is a random orthogonal rotation matrix.

    Args:
        dim: ambient dimension d
        n_modes: number of Gaussian components K
        mode_sep: distance between mode centers before rotation
        mode_std: standard deviation of each component
        seed: random seed for generating R and mode centers
        device: torch device
    """
    def __init__(
        self,
        dim: int,
        n_modes: int = 8,
        mode_sep: float = 5.0,
        mode_std: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(f"rotgmm{dim}", dim)
        self.n_modes = n_modes
        self.mode_std = mode_std
        self.device = device

        # Generate mode centers and rotation
        rng = np.random.RandomState(seed)

        # Place modes on a regular simplex-like arrangement in first m dims
        # then rotate into full d-dimensional space
        m = min(n_modes - 1, dim)  # intrinsic dimensionality of the arrangement
        centers_low = np.zeros((n_modes, dim))
        for k in range(n_modes):
            # Spread modes along first few dimensions
            angle = 2 * np.pi * k / n_modes
            if dim >= 2:
                centers_low[k, 0] = mode_sep * np.cos(angle)
                centers_low[k, 1] = mode_sep * np.sin(angle)
            if dim >= 3:
                centers_low[k, 2] = mode_sep * 0.3 * ((-1) ** k)

        # Random orthogonal rotation
        Q, _ = np.linalg.qr(rng.randn(dim, dim))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        centers_rotated = centers_low @ Q.T

        self.register_centers = torch.tensor(
            centers_rotated, dtype=torch.float32
        )
        self.register_rotation = torch.tensor(Q, dtype=torch.float32)

        # Equal weights
        self.log_weights = torch.zeros(n_modes)

        # Precompute reference samples for evaluation
        ref_samples = []
        n_per_mode = 2000 // n_modes
        for k in range(n_modes):
            samples_k = (
                torch.randn(n_per_mode, dim) * mode_std
                + torch.tensor(centers_rotated[k], dtype=torch.float32)
            )
            ref_samples.append(samples_k)
        self.ref_samples = torch.cat(ref_samples, dim=0)

    def _to_device(self, device):
        self.register_centers = self.register_centers.to(device)
        self.register_rotation = self.register_rotation.to(device)
        self.log_weights = self.log_weights.to(device)
        self.ref_samples = self.ref_samples.to(device)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x) = -log Σ_k w_k N(x; μ_k, σ²I).

        Args:
            x: (B, D) sample positions

        Returns:
            E: (B,) energies
        """
        if self.register_centers.device != x.device:
            self._to_device(x.device)

        B, D = x.shape
        K = self.n_modes

        # (B, K, D): x_i - μ_k
        diff = x.unsqueeze(1) - self.register_centers.unsqueeze(0)  # (B, K, D)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, K)

        # log N(x; μ_k, σ²I) = -||x-μ_k||²/(2σ²) - D/2 log(2πσ²)
        log_probs = -sq_dist / (2 * self.mode_std ** 2)  # (B, K)
        log_probs = log_probs + self.log_weights.unsqueeze(0)  # add log weights

        # log-sum-exp for numerical stability
        log_mixture = torch.logsumexp(log_probs, dim=1)  # (B,)

        # E = -log p (up to constant)
        return -log_mixture

    def get_ref_samples(self):
        """Return precomputed reference samples for evaluation."""
        return self.ref_samples

    def get_mode_centers(self):
        """Return the rotated mode centers."""
        return self.register_centers

    def count_modes_covered(
        self, samples: torch.Tensor, threshold_factor: float = 3.0
    ) -> dict:
        """Count how many modes are covered by the samples.

        A mode k is "covered" if at least one sample is within
        threshold_factor * mode_std of μ_k.

        Args:
            samples: (N, D)
            threshold_factor: multiples of mode_std for coverage

        Returns:
            dict with 'n_modes_covered', 'n_modes_total',
                 'coverage_fraction', 'per_mode_counts'
        """
        if self.register_centers.device != samples.device:
            self._to_device(samples.device)

        threshold = threshold_factor * self.mode_std
        centers = self.register_centers  # (K, D)

        # Distance from each sample to each center
        dists = torch.cdist(samples, centers)  # (N, K)
        min_dists = dists.min(dim=0).values  # (K,) — closest sample to each center

        covered = (min_dists < threshold)
        per_mode_counts = (dists < threshold).sum(dim=0)  # (K,) samples per mode

        return {
            'n_modes_covered': int(covered.sum().item()),
            'n_modes_total': self.n_modes,
            'coverage_fraction': covered.float().mean().item(),
            'per_mode_counts': per_mode_counts.cpu().tolist(),
            'min_dists_to_centers': min_dists.cpu().tolist(),
        }
```

### 1.4 Rotated GMM Evaluator

Add a simple evaluator for the synthetic benchmark:

```python
"""
Add to adjoint_samplers/components/evaluator.py (or a new file):

RotatedGMMEvaluator — evaluates mode coverage for rotated GMM experiments.
"""

class RotatedGMMEvaluator:
    def __init__(self, energy) -> None:
        from adjoint_samplers.energies.rotated_gmm_energy import RotatedGMMEnergy
        assert isinstance(energy, RotatedGMMEnergy)
        self.energy = energy
        self.ref_samples = energy.get_ref_samples()

    def __call__(self, samples: torch.Tensor) -> dict:
        # Mode coverage
        coverage = self.energy.count_modes_covered(samples)

        # Energy W2
        gen_E = self.energy.eval(samples)
        ref_E = self.energy.eval(self.ref_samples.to(samples.device))
        energy_w2 = pot.emd2_1d(
            ref_E.cpu().numpy(), gen_E.cpu().numpy()
        ) ** 0.5

        return {
            'n_modes_covered': coverage['n_modes_covered'],
            'n_modes_total': coverage['n_modes_total'],
            'coverage_fraction': coverage['coverage_fraction'],
            'energy_w2': energy_w2,
            'per_mode_counts': coverage['per_mode_counts'],
        }
```

### 1.5 Config Files

**Matcher config** (`configs/matcher/ksd_adjoint_ve.yaml`):

```yaml
_target_: adjoint_samplers.components.ksd_matcher.KSDAdjointVEMatcher
ksd_lambda: ${ksd_lambda}
ksd_bandwidth: null
ksd_max_particles: 2048
ksd_efficient_threshold: 1024
grad_state_cost:
  _target_: adjoint_samplers.components.state_cost.ZeroGradStateCost
buffer:
  _target_: adjoint_samplers.components.buffer.BatchBuffer
  buffer_size: ${adjoint_matcher.buffer_size}
```

**DW4 KSD config** (`configs/experiment/dw4_ksd_asbs.yaml`):

```yaml
# @package _global_
defaults:
  - /problem: dw4
  - /source: harmonic
  - /sde@ref_sde: graph_ve
  - /model@controller: egnn
  - /state_cost: zero
  - /term_cost: graph_corrector_term_cost
  - /matcher@adjoint_matcher: ksd_adjoint_ve
  - /model@corrector: egnn
  - /matcher@corrector_matcher: corrector

exp_name: dw4_ksd_asbs
scale: 2
nfe: 200
sigma_max: 1
sigma_min: 0.001
rescale_t: null
ksd_lambda: 1.0
num_epochs: 5000
max_grad_E_norm: 100
adj_num_epochs_per_stage: 200
ctr_num_epochs_per_stage: 20

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${adj_num_epochs_per_stage}
  optim:
    lr: 1e-4
    weight_decay: 0

corrector_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${ctr_num_epochs_per_stage}
  optim:
    lr: 1e-4
    weight_decay: 0

use_wandb: false
```

**LJ13 KSD config** (`configs/experiment/lj13_ksd_asbs.yaml`): Same structure, copy from `lj13_asbs.yaml` but change matcher to `ksd_adjoint_ve` and add `ksd_lambda: 1.0`.

**LJ55 KSD config** (`configs/experiment/lj55_ksd_asbs.yaml`): Same structure, copy from `lj55_asbs.yaml` but change matcher to `ksd_adjoint_ve` and add `ksd_lambda: 1.0`.

**Rotated GMM configs** — create problem configs for each dimension:

`configs/problem/rotgmm10.yaml`:

```yaml
# @package _global_
dim: 10
n_particles: 1
spatial_dim: 10

energy:
  _target_: adjoint_samplers.energies.rotated_gmm_energy.RotatedGMMEnergy
  dim: ${dim}
  n_modes: 8
  mode_sep: 5.0
  mode_std: 0.5

evaluator:
  _target_: adjoint_samplers.components.evaluator.RotatedGMMEvaluator
```

Create `rotgmm30.yaml`, `rotgmm50.yaml`, `rotgmm100.yaml` with `dim: 30`, `dim: 50`, `dim: 100` respectively. Adjust `n_modes` if desired (8 is fine for all).

For each dimension, create baseline and KSD experiment configs:

`configs/experiment/rotgmm10_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: rotgmm10
  - /source: gauss
  - /sde@ref_sde: ve        # non-graph VE since particles=1
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: term_cost    # basic GradEnergy, no corrector for AS

exp_name: rotgmm10_asbs
nfe: 100
sigma_max: 5
sigma_min: 0.01
rescale_t: null
num_epochs: 3000
max_grad_E_norm: 50

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/rotgmm10_ksd_asbs.yaml`: Same but with `ksd_adjoint_ve` matcher and `ksd_lambda: 1.0`.

Repeat for `rotgmm30`, `rotgmm50`, `rotgmm100`. For higher dimensions, adjust:

- `rotgmm30`: `nfe: 200`, `sigma_max: 8`
- `rotgmm50`: `nfe: 300`, `sigma_max: 10`
- `rotgmm100`: `nfe: 500`, `sigma_max: 15`

**Note:** The RotatedGMM is not a particle system, so it uses the non-graph `VESDE` and `FourierMLP` instead of `GraphVESDE` and `EGNN`. The source is a standard Gaussian `gauss` instead of `harmonic`. Claude Code should create appropriate SDE and term_cost configs for this non-graph setting. Check if `configs/sde/ve.yaml` and `configs/term_cost/term_cost.yaml` already exist; if not, create them pointing to the non-graph versions.

### 1.6 LJ38 — The Double-Funnel Benchmark

LJ38 is the most important new benchmark. It’s a 38-particle Lennard-Jones cluster (114 dimensions) famous for its **double-funnel energy landscape**: the global minimum is a truncated octahedron (FCC structure), but the vast majority of low-energy minima are icosahedral. No known collective variables cleanly separate these two structural families. This makes it the ideal test case for our CV-free method.

**No new energy code needed.** The existing `LennardJonesEnergy` class works with any `n_particles`. Just create a new config.

`configs/problem/lj38.yaml`:

```yaml
# @package _global_

dim: 114
n_particles: 38
spatial_dim: 3

energy:
  _target_: adjoint_samplers.energies.lennard_jones_energy.LennardJonesEnergy
  dim: ${dim}
  n_particles: ${n_particles}

evaluator:
  _target_: adjoint_samplers.components.evaluator.SyntheticEenergyEvaluator
  ref_samples_path: data/test_split_LJ38-1000.npy
```

**Reference samples for LJ38:** These are NOT included in the repo’s download script. You need to generate them by running long parallel-tempering MCMC or MD simulations. Alternatively, use samples from the Cambridge Cluster Database (Wales group) which catalogs known LJ38 minima. For a quick start, generate reference samples by running vanilla ASBS for a very long time (50k+ epochs) and saving the best samples, or use HMC with high temperature followed by annealing.

A pragmatic approach: generate 500 samples from each funnel (icosahedral and FCC) using short MD simulations initialized near known minima of each type. The icosahedral minimum has energy $\approx -173.93$ and the FCC (truncated octahedron) minimum has energy $\approx -173.25$. Store as `data/test_split_LJ38-1000.npy`.

`configs/experiment/lj38_asbs.yaml`:

```yaml
# @package _global_

defaults:
  - /problem: lj38
  - /source: harmonic
  - /sde@ref_sde: graph_ve
  - /model@controller: egnn
  - /state_cost: zero
  - /term_cost: graph_corrector_term_cost
  - /matcher@adjoint_matcher: adjoint_ve
  - /model@corrector: egnn
  - /matcher@corrector_matcher: corrector

exp_name: lj38_asbs
scale: 1
nfe: 500
sigma_max: 2
sigma_min: 0.001
rescale_t: null
num_epochs: 5000
max_grad_E_norm: 100

resample_batch_size: 256
train_batch_size: 256
eval_batch_size: 1000
num_eval_samples: 2000

adj_num_epochs_per_stage: 250
ctr_num_epochs_per_stage: 20

adjoint_matcher:
  buffer_size: 10000
  duplicates: 20
  resample_size: 256
  num_epochs_per_stage: ${adj_num_epochs_per_stage}
  optim:
    lr: 1e-4
    weight_decay: 0

corrector_matcher:
  buffer_size: 10000
  duplicates: 20
  resample_size: 256
  num_epochs_per_stage: ${ctr_num_epochs_per_stage}
  optim:
    lr: 1e-4
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/lj38_ksd_asbs.yaml`: Same as above but change matcher to `ksd_adjoint_ve` and add `ksd_lambda: 1.0`.

**Key evaluation for LJ38:** Beyond the standard W2 metrics, compute the energy of each generated sample and classify it as “icosahedral funnel” (energy < -170 and icosahedral structure) or “FCC funnel” (truncated octahedron structure). Count how many samples fall in each funnel. If baseline ASBS finds only one funnel but KSD-ASBS finds both, that’s the headline result.

### 1.7 Müller-Brown Potential — 2D Visualization Benchmark

The Müller-Brown potential is a classic 2D test case with three minima and two saddle points. It’s too low-dimensional to be challenging, but it’s perfect for **visualization** — you can plot the full energy landscape as a contour map with sample positions overlaid, showing mode coverage directly.

`adjoint_samplers/energies/muller_brown_energy.py`:

```python
"""
adjoint_samplers/energies/muller_brown_energy.py

Müller-Brown potential: 2D energy with 3 minima.
Classic test case for enhanced sampling methods.
Used here for visualization — plot samples on the 2D landscape.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class MullerBrownEnergy(BaseEnergy):
    """Müller-Brown potential in 2D.

    E(x, y) = Σ_k A_k exp(a_k(x-x0_k)² + b_k(x-x0_k)(y-y0_k) + c_k(y-y0_k)²)

    Three minima at approximately:
        (-0.558, 1.442) with E ≈ -146.7
        (0.623, 0.028)  with E ≈ -108.2
        (-0.050, 0.467) with E ≈ -80.8
    """
    # Standard Müller-Brown parameters
    A  = [-200, -100, -170, 15]
    a  = [-1, -1, -6.5, 0.7]
    b  = [0, 0, 11, 0.6]
    c  = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    def __init__(self, dim=2, device="cpu", temperature=1000.0):
        super().__init__("muller_brown", dim)
        assert dim == 2
        self.device = device
        self.temperature = temperature  # Scale factor — raw MB has huge barriers

        # Precompute reference samples via grid sampling
        self._precompute_reference()

    def _precompute_reference(self):
        """Generate reference samples by rejection sampling on a grid."""
        x = torch.linspace(-1.5, 1.2, 200)
        y = torch.linspace(-0.5, 2.0, 200)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        E = self.eval(grid)
        log_p = -E / self.temperature
        log_p = log_p - log_p.max()
        p = torch.exp(log_p)
        p = p / p.sum()

        # Resample
        idx = torch.multinomial(p, 5000, replacement=True)
        self.ref_samples = grid[idx]

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy. x: (B, 2)."""
        assert x.shape[-1] == 2
        x1 = x[..., 0]
        x2 = x[..., 1]

        result = torch.zeros_like(x1)
        for k in range(4):
            result = result + self.A[k] * torch.exp(
                self.a[k] * (x1 - self.x0[k])**2
                + self.b[k] * (x1 - self.x0[k]) * (x2 - self.y0[k])
                + self.c[k] * (x2 - self.y0[k])**2
            )
        # Scale by temperature
        return result / self.temperature

    def get_ref_samples(self):
        return self.ref_samples

    def plot_landscape(self, samples=None, samples_ksd=None, save_path=None):
        """Plot the 2D energy landscape with optional sample overlays."""
        import matplotlib.pyplot as plt

        x = torch.linspace(-1.5, 1.2, 300)
        y = torch.linspace(-0.5, 2.0, 300)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        E = self.eval(grid).reshape(300, 300)

        fig, axes = plt.subplots(1, 3 if samples_ksd is not None else 2,
                                  figsize=(6 * (3 if samples_ksd is not None else 2), 5))

        for ax in (axes if isinstance(axes, np.ndarray) else [axes]):
            ax.contourf(xx.numpy(), yy.numpy(), E.numpy(),
                       levels=50, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        axes[0].set_title('Reference Samples')
        ref = self.ref_samples[:500]
        axes[0].scatter(ref[:, 0], ref[:, 1], s=1, c='white', alpha=0.5)

        if samples is not None:
            axes[1].set_title('Baseline ASBS')
            s = samples[:500].cpu()
            axes[1].scatter(s[:, 0], s[:, 1], s=3, c='red', alpha=0.6)

        if samples_ksd is not None:
            axes[2].set_title('KSD-ASBS')
            s = samples_ksd[:500].cpu()
            axes[2].scatter(s[:, 0], s[:, 1], s=3, c='orange', alpha=0.6)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
```

`configs/problem/muller.yaml`:

```yaml
# @package _global_

dim: 2
n_particles: 1
spatial_dim: 2

energy:
  _target_: adjoint_samplers.energies.muller_brown_energy.MullerBrownEnergy
  dim: ${dim}
  temperature: 1000.0

evaluator:
  _target_: adjoint_samplers.components.evaluator.RotatedGMMEvaluator
```

`configs/experiment/muller_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: muller
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: term_cost

exp_name: muller_asbs
nfe: 100
sigma_max: 3
sigma_min: 0.01
rescale_t: null
num_epochs: 2000
max_grad_E_norm: 50

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/muller_ksd_asbs.yaml`: Same but with `ksd_adjoint_ve` matcher and `ksd_lambda: 1.0`.

**Key output:** The `plot_landscape` method produces a side-by-side contour plot showing reference samples, baseline samples, and KSD samples on the 2D energy surface. This is the most visually compelling figure in the paper — if KSD-ASBS covers all three minima while baseline covers only the deepest one, it’s immediately obvious.

### 1.8 Bayesian Logistic Regression — Non-Molecular Benchmark

This shows the method works beyond particle physics. The posterior distribution of logistic regression weights is a high-dimensional Boltzmann distribution with no obvious CVs.

`adjoint_samplers/energies/bayesian_logreg_energy.py`:

```python
"""
adjoint_samplers/energies/bayesian_logreg_energy.py

Bayesian logistic regression posterior as an energy function.
E(θ) = -Σ_i log σ(y_i θ^T x_i) + (λ/2)||θ||²

The posterior p(θ|data) ∝ exp(-E(θ)) is a high-dimensional distribution
with no known collective variables.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class BayesianLogRegEnergy(BaseEnergy):
    """Bayesian logistic regression posterior.

    Args:
        dim: number of features (+ 1 for intercept if include_intercept)
        dataset: 'australian' (d=15) or 'german' (d=25)
        prior_scale: prior std for weights (λ = 1/prior_scale²)
        device: torch device
    """
    def __init__(
        self,
        dim: int,
        dataset: str = 'australian',
        prior_scale: float = 1.0,
        device: str = 'cpu',
    ):
        super().__init__(f"blogreg_{dataset}", dim)
        self.prior_precision = 1.0 / (prior_scale ** 2)
        self.device = device

        # Load or generate data
        X, y = self._load_data(dataset, dim)
        self.X = torch.tensor(X, dtype=torch.float32)  # (N_data, dim)
        self.y = torch.tensor(y, dtype=torch.float32)   # (N_data,)
        self.N_data = self.X.shape[0]

        # Precompute reference via long HMC chain
        self.ref_samples = self._generate_reference(n_samples=2000)

    def _load_data(self, dataset, dim):
        """Load or generate classification data.

        For reproducibility without external dependencies, generate
        synthetic data that mimics the structure of real datasets.
        If sklearn is available, load the real dataset.
        """
        try:
            from sklearn.datasets import load_breast_cancer
            from sklearn.preprocessing import StandardScaler
            if dataset == 'australian':
                # Use first 15 features of breast cancer dataset
                data = load_breast_cancer()
                X = StandardScaler().fit_transform(data.data[:, :dim])
                y = 2.0 * data.target - 1.0  # Convert to {-1, +1}
            elif dataset == 'german':
                # Synthetic 25D classification
                rng = np.random.RandomState(42)
                N = 500
                X = rng.randn(N, dim)
                w_true = rng.randn(dim) * 0.5
                logits = X @ w_true
                y = 2.0 * (logits > 0).astype(float) - 1.0
                # Add label noise
                flip = rng.rand(N) < 0.1
                y[flip] *= -1
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
        except ImportError:
            # Fallback: synthetic data
            rng = np.random.RandomState(42)
            N = 500
            X = rng.randn(N, dim)
            w_true = rng.randn(dim) * 0.5
            logits = X @ w_true
            y = 2.0 * (logits > 0).astype(float) - 1.0
            flip = rng.rand(N) < 0.1
            y[flip] *= -1

        return X.astype(np.float32), y.astype(np.float32)

    def _generate_reference(self, n_samples=2000, n_steps=5000):
        """Generate reference samples via long HMC chain."""
        dim = self.dim
        theta = torch.zeros(dim)
        samples = []
        step_size = 0.01
        n_leapfrog = 20

        X = self.X
        y = self.y

        for i in range(n_steps + n_samples):
            # HMC step
            p = torch.randn(dim)
            theta_new = theta.clone()
            p_new = p.clone()

            # Leapfrog
            for _ in range(n_leapfrog):
                theta_req = theta_new.requires_grad_(True)
                E = self._eval_single(theta_req, X, y)
                g = torch.autograd.grad(E, theta_req)[0]
                p_new = p_new - 0.5 * step_size * g
                theta_new = theta_new.detach() + step_size * p_new
                theta_req = theta_new.requires_grad_(True)
                E = self._eval_single(theta_req, X, y)
                g = torch.autograd.grad(E, theta_req)[0]
                p_new = p_new - 0.5 * step_size * g
                theta_new = theta_new.detach()

            # Accept/reject
            E_old = self._eval_single(theta, X, y)
            E_new = self._eval_single(theta_new, X, y)
            dH = (E_new - E_old) + 0.5 * (p_new.norm()**2 - p.norm()**2)
            if torch.rand(1).item() < torch.exp(-dH).clamp(max=1).item():
                theta = theta_new

            if i >= n_steps:
                samples.append(theta.clone())

        return torch.stack(samples)

    def _eval_single(self, theta, X, y):
        """Compute energy for a single θ vector."""
        logits = X @ theta  # (N_data,)
        # E = -Σ log σ(y_i * logit_i) + (λ/2)||θ||²
        log_likelihood = torch.nn.functional.logsigmoid(y * logits).sum()
        log_prior = -0.5 * self.prior_precision * (theta ** 2).sum()
        return -(log_likelihood + log_prior)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for a batch. x: (B, dim)."""
        if self.X.device != x.device:
            self.X = self.X.to(x.device)
            self.y = self.y.to(x.device)

        B = x.shape[0]
        logits = x @ self.X.T  # (B, N_data) — each row is θ_b^T x_i
        # Vectorized: sum over data points
        log_lik = torch.nn.functional.logsigmoid(
            self.y.unsqueeze(0) * logits  # (B, N_data)
        ).sum(dim=1)  # (B,)
        log_prior = -0.5 * self.prior_precision * (x ** 2).sum(dim=1)
        return -(log_lik + log_prior)  # (B,)

    def get_ref_samples(self):
        return self.ref_samples
```

`configs/problem/blogreg_au.yaml`:

```yaml
# @package _global_

dim: 15
n_particles: 1
spatial_dim: 15

energy:
  _target_: adjoint_samplers.energies.bayesian_logreg_energy.BayesianLogRegEnergy
  dim: ${dim}
  dataset: australian
  prior_scale: 1.0

evaluator:
  _target_: adjoint_samplers.components.evaluator.RotatedGMMEvaluator
```

`configs/problem/blogreg_ge.yaml`: Same with `dim: 25` and `dataset: german`.

`configs/experiment/blogreg_au_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: blogreg_au
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: term_cost

exp_name: blogreg_au_asbs
nfe: 100
sigma_max: 3
sigma_min: 0.01
rescale_t: null
num_epochs: 2000
max_grad_E_norm: 50

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/blogreg_au_ksd_asbs.yaml`: Same with `ksd_adjoint_ve` and `ksd_lambda: 1.0`.

Repeat for `blogreg_ge_asbs.yaml` and `blogreg_ge_ksd_asbs.yaml`.

### 1.9 New Training Phases for Additional Benchmarks

Add to Phase 2 baselines script:

```bash
# LJ38 baselines (3 seeds, ~8 hrs each)
echo "--- LJ38 Baselines ---"
for SEED in 0 1 2; do
  python train.py experiment=lj38_asbs seed=${SEED} use_wandb=false exp_name=lj38_asbs_s${SEED}
done

# Müller-Brown baselines (3 seeds, ~10 min each)
echo "--- Müller-Brown Baselines ---"
for SEED in 0 1 2; do
  python train.py experiment=muller_asbs seed=${SEED} use_wandb=false exp_name=muller_asbs_s${SEED}
done

# Bayesian LogReg baselines (2 datasets × 3 seeds, ~20 min each)
echo "--- Bayesian LogReg Baselines ---"
for DATASET in au ge; do
  for SEED in 0 1 2; do
    python train.py experiment=blogreg_${DATASET}_asbs seed=${SEED} use_wandb=false \
      exp_name=blogreg_${DATASET}_asbs_s${SEED}
  done
done
```

Add to Phase 3 KSD script:

```bash
# LJ38 KSD (3 λ × 3 seeds)
echo "--- LJ38 KSD ---"
for LAMBDA in 0.5 1.0 5.0; do
  for SEED in 0 1 2; do
    python train.py experiment=lj38_ksd_asbs seed=${SEED} ksd_lambda=${LAMBDA} \
      use_wandb=false exp_name=lj38_ksd_l${LAMBDA}_s${SEED}
  done
done

# Müller-Brown KSD (3 seeds)
echo "--- Müller-Brown KSD ---"
for SEED in 0 1 2; do
  python train.py experiment=muller_ksd_asbs seed=${SEED} ksd_lambda=1.0 \
    use_wandb=false exp_name=muller_ksd_s${SEED}
done

# Bayesian LogReg KSD (2 datasets × 3 seeds)
echo "--- Bayesian LogReg KSD ---"
for DATASET in au ge; do
  for SEED in 0 1 2; do
    python train.py experiment=blogreg_${DATASET}_ksd_asbs seed=${SEED} ksd_lambda=1.0 \
      use_wandb=false exp_name=blogreg_${DATASET}_ksd_s${SEED}
  done
done
```

### 1.10 Updated Priority Order

If compute is limited, prioritize:

1. **DW4 baseline + KSD** (3 hrs) — proves the concept
1. **Müller-Brown baseline + KSD** (30 min) — produces the best visualization figure
1. **Rotated GMM d=10,30** (2 hrs) — proves synthetic CV-unknown advantage
1. **DW4 λ ablation** (5 hrs) — finds optimal hyperparameter
1. **LJ38 baseline + KSD** (24 hrs) — the headline real-system result (double funnel)
1. **Bayesian LogReg** (2 hrs) — proves it works beyond molecular systems
1. **LJ13 baseline + KSD** (24 hrs) — medium-scale molecular
1. **Rotated GMM d=50,100** (2 hrs) — dimension scaling
1. **LJ55 baseline + KSD** (72 hrs) — only if LJ38 works

-----

## Phase 2: Baseline Training

### 2.1 Training Script (`scripts/run_phase2_baselines.sh`)

```bash
#!/bin/bash
# Phase 2: Train all baselines
# Estimated time: ~50 GPU-hours total

set -e

echo "=== Phase 2: Baseline ASBS Training ==="

# Download reference samples
bash scripts/download.sh

# DW4 baselines (3 seeds, ~1 hr each)
echo "--- DW4 Baselines ---"
python train.py experiment=dw4_asbs seed=0 use_wandb=false exp_name=dw4_asbs_s0
python train.py experiment=dw4_asbs seed=1 use_wandb=false exp_name=dw4_asbs_s1
python train.py experiment=dw4_asbs seed=2 use_wandb=false exp_name=dw4_asbs_s2

# LJ13 baselines (3 seeds, ~4 hrs each)
echo "--- LJ13 Baselines ---"
python train.py experiment=lj13_asbs seed=0 use_wandb=false exp_name=lj13_asbs_s0
python train.py experiment=lj13_asbs seed=1 use_wandb=false exp_name=lj13_asbs_s1
python train.py experiment=lj13_asbs seed=2 use_wandb=false exp_name=lj13_asbs_s2

# LJ55 baselines (3 seeds, ~12 hrs each)
echo "--- LJ55 Baselines ---"
python train.py experiment=lj55_asbs seed=0 use_wandb=false exp_name=lj55_asbs_s0
python train.py experiment=lj55_asbs seed=1 use_wandb=false exp_name=lj55_asbs_s1
python train.py experiment=lj55_asbs seed=2 use_wandb=false exp_name=lj55_asbs_s2

echo "=== Phase 2 Complete ==="
```

-----

## Phase 3: KSD-ASBS Training

### 3.1 λ Ablation on DW4 (`scripts/run_phase3_ksd.sh`)

```bash
#!/bin/bash
# Phase 3: KSD-augmented ASBS training with λ ablation
set -e

echo "=== Phase 3: KSD-ASBS Training ==="

# --- DW4 λ ablation (5 λ × 3 seeds = 15 runs) ---
echo "--- DW4 KSD λ Ablation ---"
for LAMBDA in 0.1 0.5 1.0 5.0 10.0; do
  for SEED in 0 1 2; do
    echo "DW4: λ=${LAMBDA}, seed=${SEED}"
    python train.py experiment=dw4_ksd_asbs \
      seed=${SEED} \
      ksd_lambda=${LAMBDA} \
      use_wandb=false \
      exp_name=dw4_ksd_l${LAMBDA}_s${SEED}
  done
done

# --- LJ13 (best λ from DW4 ablation + neighbors, 3 λ × 3 seeds = 9 runs) ---
echo "--- LJ13 KSD ---"
for LAMBDA in 0.5 1.0 5.0; do
  for SEED in 0 1 2; do
    echo "LJ13: λ=${LAMBDA}, seed=${SEED}"
    python train.py experiment=lj13_ksd_asbs \
      seed=${SEED} \
      ksd_lambda=${LAMBDA} \
      use_wandb=false \
      exp_name=lj13_ksd_l${LAMBDA}_s${SEED}
  done
done

# --- LJ55 (best λ only, 1 λ × 3 seeds = 3 runs) ---
echo "--- LJ55 KSD ---"
BEST_LAMBDA=1.0  # Update after DW4/LJ13 ablation
for SEED in 0 1 2; do
  echo "LJ55: λ=${BEST_LAMBDA}, seed=${SEED}"
  python train.py experiment=lj55_ksd_asbs \
    seed=${SEED} \
    ksd_lambda=${BEST_LAMBDA} \
    use_wandb=false \
    exp_name=lj55_ksd_l${BEST_LAMBDA}_s${SEED}
done

echo "=== Phase 3 Complete ==="
```

### 3.2 Chunking Ablation

To test whether chunking affects results (it shouldn’t) and measure wall-clock time:

```bash
# Chunking comparison on DW4 (fixed λ=1.0, seed=0)
# Vary ksd_efficient_threshold to force chunking or full computation

# Full computation (no chunking)
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=1.0 \
  use_wandb=false exp_name=dw4_ksd_chunk_full \
  adjoint_matcher.ksd_efficient_threshold=99999

# Chunked (chunk_size=128)
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=1.0 \
  use_wandb=false exp_name=dw4_ksd_chunk_128 \
  adjoint_matcher.ksd_efficient_threshold=0

# Compare: training loss curves should be identical (up to floating point)
# Wall-clock time difference shows chunking overhead
```

### 3.3 Batch Size Ablation

To test how batch size (N particles per buffer refresh) affects KSD correction quality:

```bash
# Batch size comparison on DW4 (fixed λ=1.0, seed=0)
for BSIZE in 64 128 256 512 1024; do
  python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=1.0 \
    resample_batch_size=${BSIZE} \
    use_wandb=false exp_name=dw4_ksd_bsize${BSIZE}
done
```

-----

## Phase 4: Synthetic Experiments (CV-Unknown)

### 4.1 Training Script (`scripts/run_phase4_synthetic.sh`)

```bash
#!/bin/bash
# Phase 4: Rotated GMM experiments — demonstrating advantage when CVs are unknown
set -e

echo "=== Phase 4: Synthetic CV-Unknown Experiments ==="

# For each dimension: train baseline ASBS and KSD-ASBS
for DIM in 10 30 50 100; do
  for SEED in 0 1 2; do
    echo "RotGMM d=${DIM}: Baseline, seed=${SEED}"
    python train.py experiment=rotgmm${DIM}_asbs \
      seed=${SEED} use_wandb=false \
      exp_name=rotgmm${DIM}_asbs_s${SEED}

    echo "RotGMM d=${DIM}: KSD, seed=${SEED}"
    python train.py experiment=rotgmm${DIM}_ksd_asbs \
      seed=${SEED} ksd_lambda=1.0 use_wandb=false \
      exp_name=rotgmm${DIM}_ksd_s${SEED}
  done
done

echo "=== Phase 4 Complete ==="
```

-----

## Phase 5: Comprehensive Evaluation

### 5.1 Evaluation Script (`evaluate_all.py`)

This is the master evaluation script that loads every checkpoint and computes all metrics.

```python
"""
evaluate_all.py

Master evaluation script. Loads all trained checkpoints from Phases 2–4,
generates samples, computes every metric, saves structured results.

Usage:
    python evaluate_all.py --outputs_root outputs --results_dir results

Expects checkpoints at:
    outputs/{exp_name}/checkpoints/checkpoint_latest.pt
    outputs/{exp_name}/config.yaml
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import OmegaConf
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import (
    compute_ksd_squared, median_bandwidth,
    compute_stein_kernel_gradient, compute_stein_kernel_gradient_efficient,
)
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Metric computation
# ====================================================================

@torch.no_grad()
def compute_all_metrics(
    samples: torch.Tensor,
    energy,
    ref_samples: torch.Tensor = None,
    n_particles: int = None,
    spatial_dim: int = None,
    is_gmm: bool = False,
) -> dict:
    """Compute every metric for a set of terminal samples."""
    metrics = {}
    device = samples.device
    N, D = samples.shape

    # --- Energy statistics ---
    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

    # --- KSD ---
    scores = energy.score(samples)
    ell = median_bandwidth(samples)
    N_ksd = min(N, 2000)
    idx = torch.randperm(N, device=device)[:N_ksd]
    metrics['ksd_squared'] = compute_ksd_squared(
        samples[idx], scores[idx], ell
    ).item()
    metrics['bandwidth'] = ell.item()

    # --- Reference-based metrics ---
    if ref_samples is not None:
        ref_samples = ref_samples.to(device)
        B = min(N, len(ref_samples))
        idx_g = torch.randperm(N, device=device)[:B]
        idx_r = torch.randperm(len(ref_samples), device=device)[:B]
        gen = samples[idx_g]
        ref = ref_samples[idx_r]

        # Energy W2
        ref_E = energy.eval(ref)
        metrics['ref_mean_energy'] = ref_E.mean().item()
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E.cpu().numpy(), gen_E[idx_g].cpu().numpy()) ** 0.5
        )

        # Particle-system metrics
        if n_particles is not None and n_particles > 1:
            gen_dist = interatomic_dist(gen, n_particles, spatial_dim)
            ref_dist = interatomic_dist(ref, n_particles, spatial_dim)
            metrics['dist_w2'] = float(pot.emd2_1d(
                gen_dist.cpu().numpy().reshape(-1),
                ref_dist.cpu().numpy().reshape(-1),
            ))

            M = dist_point_clouds(
                gen.reshape(-1, n_particles, spatial_dim).cpu(),
                ref.reshape(-1, n_particles, spatial_dim).cpu(),
            )
            a = torch.ones(M.shape[0]) / M.shape[0]
            b = torch.ones(M.shape[0]) / M.shape[0]
            metrics['eq_w2'] = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)

    # --- GMM-specific: mode coverage ---
    if is_gmm and hasattr(energy, 'count_modes_covered'):
        cov = energy.count_modes_covered(samples)
        metrics['n_modes_covered'] = cov['n_modes_covered']
        metrics['n_modes_total'] = cov['n_modes_total']
        metrics['coverage_fraction'] = cov['coverage_fraction']
        metrics['per_mode_counts'] = cov['per_mode_counts']

    # --- Store energy and score histograms (as arrays for plotting later) ---
    metrics['_energy_values'] = gen_E.cpu().numpy().tolist()

    return metrics


def compute_chunking_timing(samples, scores, ell):
    """Compare wall-clock time of full vs chunked Stein kernel gradient."""
    N = samples.shape[0]
    timings = {}

    # Full
    torch.cuda.synchronize()
    t0 = time.time()
    g_full = compute_stein_kernel_gradient(samples, scores, ell)
    torch.cuda.synchronize()
    timings['full_time'] = time.time() - t0

    # Chunked (chunk=128)
    torch.cuda.synchronize()
    t0 = time.time()
    g_chunk = compute_stein_kernel_gradient_efficient(samples, scores, ell, chunk_size=128)
    torch.cuda.synchronize()
    timings['chunk128_time'] = time.time() - t0

    # Chunked (chunk=256)
    torch.cuda.synchronize()
    t0 = time.time()
    g_chunk2 = compute_stein_kernel_gradient_efficient(samples, scores, ell, chunk_size=256)
    torch.cuda.synchronize()
    timings['chunk256_time'] = time.time() - t0

    # Verify equivalence
    timings['max_diff_chunk128'] = (g_full - g_chunk).abs().max().item()
    timings['max_diff_chunk256'] = (g_full - g_chunk2).abs().max().item()

    return timings


# ====================================================================
# Experiment loading and sample generation
# ====================================================================

def load_experiment(exp_name, outputs_root, device):
    """Load a trained experiment: config, model, energy, source."""
    exp_dir = Path(outputs_root) / exp_name
    cfg_path = exp_dir / 'config.yaml'
    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'

    if not cfg_path.exists():
        # Try Hydra's .hydra directory
        cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not ckpt_path.exists():
        print(f"  WARNING: No checkpoint found for {exp_name}")
        return None

    cfg = OmegaConf.load(cfg_path)
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    ts_cfg = {
        't0': cfg.timesteps.t0, 't1': cfg.timesteps.t1,
        'steps': cfg.timesteps.steps, 'rescale_t': cfg.timesteps.rescale_t,
    }

    return {
        'sde': sde, 'source': source, 'energy': energy,
        'ts_cfg': ts_cfg, 'cfg': cfg,
    }


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, batch_size, device):
    """Generate terminal samples."""
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1)
        n += b
    return torch.cat(x1_list)[:n_samples]


# ====================================================================
# Main evaluation loop
# ====================================================================

def evaluate_experiment_group(
    exp_names: list,
    outputs_root: str,
    device: str,
    n_samples: int = 2000,
    n_eval_seeds: int = 5,
) -> dict:
    """Evaluate a group of experiments (same benchmark, different methods/seeds)."""
    results = {}

    for exp_name in exp_names:
        print(f"\n  Evaluating {exp_name}...")
        exp = load_experiment(exp_name, outputs_root, device)
        if exp is None:
            continue

        cfg = exp['cfg']
        energy = exp['energy']
        is_gmm = 'rotgmm' in exp_name

        # Determine particle system properties
        n_particles = getattr(energy, 'n_particles', 1)
        spatial_dim = getattr(energy, 'n_spatial_dim', cfg.get('dim', None))

        # Load reference samples
        ref_samples = None
        if hasattr(cfg, 'evaluator') and 'ref_samples_path' in cfg.evaluator:
            import os
            root = Path(os.path.abspath(__file__)).parent
            ref_path = root / cfg.evaluator.ref_samples_path
            if ref_path.exists():
                ref_np = np.load(ref_path, allow_pickle=True)
                ref_samples = torch.tensor(ref_np, dtype=torch.float32)
                if n_particles > 1:
                    ref_samples = remove_mean(ref_samples, n_particles, spatial_dim)
        elif is_gmm and hasattr(energy, 'get_ref_samples'):
            ref_samples = energy.get_ref_samples()

        # Multiple evaluation seeds
        seed_metrics = []
        batch_size = min(n_samples, cfg.get('eval_batch_size', 2000))

        for eval_seed in range(n_eval_seeds):
            torch.manual_seed(eval_seed * 7777)
            samples = generate_samples(
                exp['sde'], exp['source'], exp['ts_cfg'],
                n_samples, batch_size, device
            )
            m = compute_all_metrics(
                samples, energy, ref_samples,
                n_particles if n_particles > 1 else None,
                spatial_dim if n_particles > 1 else None,
                is_gmm=is_gmm,
            )
            seed_metrics.append(m)

        # Aggregate across eval seeds
        agg = {}
        for key in seed_metrics[0]:
            if key.startswith('_'):
                agg[key] = seed_metrics[0][key]  # Store first seed's raw data
                continue
            if isinstance(seed_metrics[0][key], (int, float)):
                vals = [m[key] for m in seed_metrics]
                agg[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
            elif isinstance(seed_metrics[0][key], list):
                agg[key] = seed_metrics[0][key]  # Lists: take first seed

        results[exp_name] = agg

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_root', type=str, default='outputs')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--n_eval_seeds', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover all experiments
    outputs_root = Path(args.outputs_root)
    all_experiments = sorted([
        d.name for d in outputs_root.iterdir()
        if d.is_dir() and (d / 'checkpoints').exists()
    ])
    print(f"Found {len(all_experiments)} experiments: {all_experiments}")

    # Group by benchmark
    groups = defaultdict(list)
    for exp in all_experiments:
        if exp.startswith('dw4'):
            groups['dw4'].append(exp)
        elif exp.startswith('lj13'):
            groups['lj13'].append(exp)
        elif exp.startswith('lj55'):
            groups['lj55'].append(exp)
        elif exp.startswith('rotgmm'):
            dim = exp.split('_')[0]  # e.g., 'rotgmm10'
            groups[dim].append(exp)
        else:
            groups['other'].append(exp)

    # Evaluate each group
    all_results = {}
    for group_name, exp_names in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating group: {group_name} ({len(exp_names)} experiments)")
        print(f"{'='*60}")

        group_results = evaluate_experiment_group(
            exp_names, str(outputs_root), device,
            args.n_samples, args.n_eval_seeds,
        )
        all_results[group_name] = group_results

        # Save per-group results
        group_path = results_dir / f'{group_name}_results.json'
        with open(group_path, 'w') as f:
            json.dump(group_results, f, indent=2, default=str)
        print(f"  Saved to {group_path}")

    # --- Chunking timing test ---
    print(f"\n{'='*60}")
    print("Chunking timing test")
    print(f"{'='*60}")
    timing_results = {}
    for dim_label, dim in [('dw4_8d', 8), ('lj13_39d', 39), ('lj55_165d', 165)]:
        N = 512
        samples = torch.randn(N, dim, device=device)
        scores = -samples  # dummy scores
        ell = median_bandwidth(samples)
        t = compute_chunking_timing(samples, scores, ell)
        timing_results[dim_label] = t
        print(f"  {dim_label}: full={t['full_time']:.4f}s, "
              f"chunk128={t['chunk128_time']:.4f}s, "
              f"chunk256={t['chunk256_time']:.4f}s, "
              f"max_diff={t['max_diff_chunk128']:.2e}")

    with open(results_dir / 'chunking_timing.json', 'w') as f:
        json.dump(timing_results, f, indent=2)

    # Save master results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n=== All evaluation complete. Results in {results_dir}/ ===")


if __name__ == '__main__':
    main()
```

### 5.2 Running Evaluation (`scripts/run_phase5_evaluate.sh`)

```bash
#!/bin/bash
set -e
echo "=== Phase 5: Comprehensive Evaluation ==="
python evaluate_all.py \
  --outputs_root outputs \
  --results_dir results \
  --n_samples 2000 \
  --n_eval_seeds 5
echo "=== Phase 5 Complete ==="
```

-----

## Phase 6: Results Generation

### 6.1 Results Generator (`generate_results.py`)

This script reads all JSON results files and produces `RESULTS.md` with every table and figure.

```python
"""
generate_results.py

Reads results/ directory and generates docs/RESULTS.md with:
- Per-benchmark comparison tables (baseline vs KSD-ASBS)
- λ ablation tables and plots
- Batch size ablation
- Chunking timing table
- Synthetic rotated GMM mode coverage
- Energy histogram overlays
- Dimension scaling plots
- Training cost comparison

Usage:
    python generate_results.py --results_dir results --output docs/RESULTS.md
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_all_results(results_dir):
    """Load all JSON result files."""
    data = {}
    for f in Path(results_dir).glob('*_results.json'):
        group = f.stem.replace('_results', '')
        with open(f) as fh:
            data[group] = json.load(fh)
    # Chunking timing
    timing_path = Path(results_dir) / 'chunking_timing.json'
    if timing_path.exists():
        with open(timing_path) as f:
            data['_chunking'] = json.load(f)
    return data


def fmt(val, key=''):
    """Format a metric value."""
    if isinstance(val, dict):
        m, s = val.get('mean', 0), val.get('std', 0)
        if abs(m) < 0.01:
            return f"{m:.2e} ± {s:.2e}"
        return f"{m:.4f} ± {s:.4f}"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def make_benchmark_table(group_data, benchmark, metrics, fig_dir):
    """Generate markdown table + energy histogram for one benchmark."""
    lines = []
    lines.append(f"### {benchmark.upper()}")
    lines.append("")

    # Find baseline and KSD experiments
    baseline_exps = {k: v for k, v in group_data.items() if 'ksd' not in k}
    ksd_exps = {k: v for k, v in group_data.items() if 'ksd' in k}

    if not baseline_exps or not ksd_exps:
        lines.append("_Incomplete data — missing baseline or KSD experiments._")
        return "\n".join(lines)

    # Aggregate across seeds for baseline
    def aggregate_seeds(exps):
        agg = defaultdict(list)
        for exp_name, exp_data in exps.items():
            for metric in metrics:
                if metric in exp_data and isinstance(exp_data[metric], dict):
                    agg[metric].extend(exp_data[metric].get('values', []))
        result = {}
        for metric, vals in agg.items():
            result[metric] = {'mean': np.mean(vals), 'std': np.std(vals)}
        return result

    base_agg = aggregate_seeds(baseline_exps)
    ksd_agg = aggregate_seeds(ksd_exps)

    # Table
    lines.append("| Metric | Baseline ASBS | KSD-ASBS | Δ (%) |")
    lines.append("|---|---|---|---|")
    for metric in metrics:
        if metric not in base_agg or metric not in ksd_agg:
            continue
        bm = base_agg[metric]['mean']
        km = ksd_agg[metric]['mean']
        delta = ((km - bm) / (abs(bm) + 1e-10)) * 100
        better = "✓" if (km < bm and 'coverage' not in metric) or \
                        (km > bm and 'coverage' in metric) else ""
        lines.append(
            f"| {metric} | {fmt(base_agg[metric])} | {fmt(ksd_agg[metric])} | "
            f"{delta:+.1f}% {better} |"
        )
    lines.append("")

    # Energy histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    # Get energy values from first experiment of each type
    for exp_name, exp_data in baseline_exps.items():
        if '_energy_values' in exp_data:
            E = exp_data['_energy_values']
            if isinstance(E, list) and len(E) > 0:
                ax.hist(E, bins=50, alpha=0.5, density=True,
                        label='Baseline ASBS', color='#1f77b4')
            break
    for exp_name, exp_data in ksd_exps.items():
        if '_energy_values' in exp_data:
            E = exp_data['_energy_values']
            if isinstance(E, list) and len(E) > 0:
                ax.hist(E, bins=50, alpha=0.5, density=True,
                        label='KSD-ASBS', color='#ff7f0e')
            break

    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
    ax.set_title(f'{benchmark.upper()}: Energy Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = fig_dir / f'{benchmark}_energy_hist.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    lines.append(f"![{benchmark} energy histogram](figures/{benchmark}_energy_hist.png)")
    lines.append("")

    return "\n".join(lines)


def make_lambda_ablation(group_data, benchmark, fig_dir):
    """Generate λ ablation table and plot."""
    lines = []
    lines.append(f"### {benchmark.upper()} — λ Ablation")
    lines.append("")

    # Parse λ from experiment names
    lambda_data = {}
    for exp_name, exp_data in group_data.items():
        if 'ksd_l' in exp_name:
            # Extract λ: e.g., "dw4_ksd_l0.5_s0" → 0.5
            parts = exp_name.split('_l')
            if len(parts) > 1:
                lam_str = parts[1].split('_')[0]
                try:
                    lam = float(lam_str)
                    if lam not in lambda_data:
                        lambda_data[lam] = defaultdict(list)
                    for metric, val in exp_data.items():
                        if isinstance(val, dict) and 'mean' in val:
                            lambda_data[lam][metric].append(val['mean'])
                except ValueError:
                    pass

    if not lambda_data:
        lines.append("_No λ ablation data found._")
        return "\n".join(lines)

    # Table
    lambdas = sorted(lambda_data.keys())
    key_metrics = ['energy_w2', 'ksd_squared', 'mean_energy']
    available = [m for m in key_metrics if m in lambda_data[lambdas[0]]]

    header = "| λ | " + " | ".join(available) + " |"
    sep = "|---|" + "|".join(["---"] * len(available)) + "|"
    lines.append(header)
    lines.append(sep)
    for lam in lambdas:
        row = f"| {lam} |"
        for metric in available:
            vals = lambda_data[lam][metric]
            m, s = np.mean(vals), np.std(vals)
            row += f" {m:.4f} ± {s:.4f} |"
        lines.append(row)
    lines.append("")

    # Plot
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        means = [np.mean(lambda_data[l][metric]) for l in lambdas]
        stds = [np.std(lambda_data[l][metric]) for l in lambdas]
        ax.errorbar(lambdas, means, yerr=stds, marker='o', capsize=3, linewidth=2)
        ax.set_xlabel('λ (KSD weight)')
        ax.set_ylabel(metric)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
    fig.suptitle(f'{benchmark.upper()}: λ Ablation')
    fig.tight_layout()
    fig_path = fig_dir / f'{benchmark}_lambda_ablation.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    lines.append(f"![{benchmark} λ ablation](figures/{benchmark}_lambda_ablation.png)")
    lines.append("")

    return "\n".join(lines)


def make_mode_coverage_table(all_data, fig_dir):
    """Generate mode coverage comparison for rotated GMM experiments."""
    lines = []
    lines.append("## Synthetic Benchmark: Rotated Gaussian Mixture (CV-Unknown)")
    lines.append("")
    lines.append("These experiments test mode coverage on energy functions where")
    lines.append("collective variables are unknown by construction (the modes are")
    lines.append("separated along a randomly rotated axis).")
    lines.append("")

    dims = []
    for key in sorted(all_data.keys()):
        if key.startswith('rotgmm'):
            dims.append(key)

    if not dims:
        lines.append("_No rotated GMM experiments found._")
        return "\n".join(lines)

    # Collect coverage data
    coverage_data = {'baseline': {}, 'ksd': {}}
    for dim_key in dims:
        dim_num = dim_key.replace('rotgmm', '')
        group = all_data[dim_key]
        for exp_name, exp_data in group.items():
            if 'coverage_fraction' in exp_data:
                cf = exp_data['coverage_fraction']
                val = cf['mean'] if isinstance(cf, dict) else cf
                if 'ksd' in exp_name:
                    coverage_data['ksd'][dim_num] = val
                else:
                    coverage_data['baseline'][dim_num] = val

    # Table
    lines.append("| Dimension | Baseline Coverage | KSD Coverage | Δ |")
    lines.append("|---|---|---|---|")
    dim_nums = sorted(set(list(coverage_data['baseline'].keys()) +
                          list(coverage_data['ksd'].keys())),
                      key=lambda x: int(x))
    for d in dim_nums:
        bc = coverage_data['baseline'].get(d, float('nan'))
        kc = coverage_data['ksd'].get(d, float('nan'))
        delta = kc - bc if not (np.isnan(bc) or np.isnan(kc)) else float('nan')
        lines.append(f"| d={d} | {bc:.2%} | {kc:.2%} | {delta:+.2%} |")
    lines.append("")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x_dims = [int(d) for d in dim_nums]
    bc_vals = [coverage_data['baseline'].get(d, 0) for d in dim_nums]
    kc_vals = [coverage_data['ksd'].get(d, 0) for d in dim_nums]
    w = 0.35
    x = np.arange(len(x_dims))
    ax.bar(x - w/2, bc_vals, w, label='Baseline ASBS', color='#1f77b4')
    ax.bar(x + w/2, kc_vals, w, label='KSD-ASBS', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in x_dims])
    ax.set_ylabel('Mode Coverage Fraction')
    ax.set_title('Mode Coverage vs Dimension (Rotated GMM)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig_path = fig_dir / 'rotgmm_coverage.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    lines.append(f"![Mode coverage](figures/rotgmm_coverage.png)")
    lines.append("")

    return "\n".join(lines)


def make_chunking_table(chunking_data):
    """Generate chunking timing comparison table."""
    lines = []
    lines.append("## Chunking Analysis")
    lines.append("")
    lines.append("Wall-clock time for computing the Stein kernel gradient sum")
    lines.append("(N=512 particles). Chunking is mathematically equivalent")
    lines.append("(max absolute difference shown).")
    lines.append("")
    lines.append("| Dimension | Full (s) | Chunk-128 (s) | Chunk-256 (s) | Slowdown | Max Diff |")
    lines.append("|---|---|---|---|---|---|")

    for dim_label, t in chunking_data.items():
        slowdown = t['chunk128_time'] / (t['full_time'] + 1e-10)
        lines.append(
            f"| {dim_label} | {t['full_time']:.4f} | {t['chunk128_time']:.4f} | "
            f"{t['chunk256_time']:.4f} | {slowdown:.1f}× | {t['max_diff_chunk128']:.2e} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_results_md(args):
    """Generate the complete RESULTS.md."""
    results_dir = Path(args.results_dir)
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all_results(results_dir)

    lines = []
    lines.append("# Results: KSD-Augmented ASBS — Comprehensive Evaluation")
    lines.append("")
    lines.append("*Auto-generated by `generate_results.py`*")
    lines.append("")

    # --- Summary ---
    lines.append("## Method")
    lines.append("")
    lines.append("KSD-Augmented ASBS modifies the adjoint terminal condition:")
    lines.append("")
    lines.append("$$Y_1^i = -\\frac{1}{N}\\nabla\\Phi_0(X_1^i) - "
                 "\\frac{\\lambda}{N^2}\\sum_j \\nabla_x k_p(X_1^i, X_1^j)$$")
    lines.append("")
    lines.append("See `docs/STEIN_VARIATIONAL_ASBS.md` for derivation.")
    lines.append("")

    # --- Per-benchmark results ---
    particle_metrics = ['energy_w2', 'dist_w2', 'eq_w2', 'ksd_squared', 'mean_energy']
    simple_metrics = ['energy_w2', 'ksd_squared', 'mean_energy']

    lines.append("## Molecular Benchmarks")
    lines.append("")
    for benchmark in ['dw4', 'lj13', 'lj55']:
        if benchmark in all_data:
            metrics = particle_metrics if benchmark != 'rotgmm' else simple_metrics
            lines.append(make_benchmark_table(
                all_data[benchmark], benchmark, metrics, fig_dir
            ))

    # --- λ ablation ---
    lines.append("## λ Ablation")
    lines.append("")
    for benchmark in ['dw4', 'lj13']:
        if benchmark in all_data:
            lines.append(make_lambda_ablation(all_data[benchmark], benchmark, fig_dir))

    # --- Synthetic experiments ---
    lines.append(make_mode_coverage_table(all_data, fig_dir))

    # --- Chunking ---
    if '_chunking' in all_data:
        lines.append(make_chunking_table(all_data['_chunking']))

    # --- Conclusions ---
    lines.append("## Conclusions")
    lines.append("")
    lines.append("*(To be written based on experimental results)*")
    lines.append("")
    lines.append("Key questions answered:")
    lines.append("")
    lines.append("1. Does KSD-ASBS reduce mode collapse compared to baseline ASBS?")
    lines.append("2. What is the optimal λ?")
    lines.append("3. Does the advantage persist in high dimensions?")
    lines.append("4. Does KSD-ASBS work where CVs are unknown (rotated GMM)?")
    lines.append("5. What is the computational overhead?")
    lines.append("6. Is chunking mathematically equivalent and practically efficient?")

    # --- Reproduction ---
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/run_phase2_baselines.sh  # Train baselines")
    lines.append("bash scripts/run_phase3_ksd.sh        # Train KSD-ASBS")
    lines.append("bash scripts/run_phase4_synthetic.sh   # Train on rotated GMM")
    lines.append("bash scripts/run_phase5_evaluate.sh    # Evaluate everything")
    lines.append("python generate_results.py             # Generate this report")
    lines.append("```")

    # Write
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write("\n".join(lines))
    print(f"RESULTS.md written to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--output', default='docs/RESULTS.md')
    args = parser.parse_args()
    generate_results_md(args)
```

-----

## Phase 7: Execution Order

### Step-by-step for Claude Code:

```
1. Create all source files (Phase 1)
   - stein_kernel.py → test it
   - ksd_matcher.py
   - rotated_gmm_energy.py
   - All config YAMLs
   - Verify: python -m adjoint_samplers.components.stein_kernel

2. Train DW4 baselines (fastest, ~1 hr each)
   - 3 seeds of dw4_asbs

3. Train DW4 KSD with λ ablation (fast iteration)
   - 5 λ values × 3 seeds
   - Check: does training converge? If loss diverges for λ=10, remove it

4. Quick evaluation of DW4 results
   - Run evaluate_all.py on DW4 experiments only
   - Check energy histograms: does KSD-ASBS show broader coverage?
   - Determine best λ for DW4

5. Train LJ13 (medium, ~4 hrs each)
   - 3 seeds baseline + 3 λ values × 3 seeds KSD
   - Use best λ from DW4 ± neighbors

6. Train synthetic rotated GMM (fast, ~20 min each)
   - d=10, 30, 50, 100 × baseline + KSD × 3 seeds
   - This is the "CV-unknown" advantage experiment

7. Train LJ55 (slow, ~12 hrs each)
   - Only best λ × 3 seeds
   - Only if DW4/LJ13 showed improvement

8. Full evaluation
   - python evaluate_all.py over everything

9. Generate report
   - python generate_results.py
   - Review docs/RESULTS.md
```

### Priority order if compute is limited:

1. **DW4 baseline + KSD** (3 hrs) — proves the method works
1. **Rotated GMM d=10,30** (2 hrs) — proves the CV-unknown advantage
1. **DW4 λ ablation** (5 hrs) — finds optimal hyperparameter
1. **LJ13 baseline + KSD** (24 hrs) — tests scaling to 39D
1. **Rotated GMM d=50,100** (2 hrs) — tests scaling for synthetics
1. **LJ55 baseline + KSD** (72 hrs) — tests scaling to 165D (only if prior results positive)

-----

## Important Notes for Claude Code

1. **Config system**: Hydra creates output directories automatically at `outputs/{exp_name}/`. Checkpoints go to `outputs/{exp_name}/checkpoints/`. Config is saved at `outputs/{exp_name}/.hydra/config.yaml` or `outputs/{exp_name}/config.yaml`.
1. **Non-graph models for RotGMM**: The rotated GMM is NOT a particle system. It uses `FourierMLP` (not `EGNN`), standard `VESDE` (not `GraphVESDE`), standard `Gauss` source (not `harmonic`), and `GradEnergy` terminal cost (not `GraphCorrectorGradTermCost`). Create appropriate non-graph configs. Check if `configs/sde/ve.yaml` exists; if not, create it pointing to `adjoint_samplers.components.sde.VESDE`.
1. **RotGMM has n_particles=1**: The RotGMM energy pretends to be a 1-particle system with `spatial_dim=dim`. This sidesteps the graph machinery. Alternatively, implement it without the particle abstraction — but the evaluator/training pipeline expects `n_particles` and `spatial_dim` fields.
1. **Memory on LJ55**: With `resample_batch_size=200` (LJ55 default) and `dim=165`, the Stein kernel gradient tensor is `200×200×165×4 bytes ≈ 26 MB` — well within GPU memory. The efficient chunked version is not needed at this batch size.
1. **Training time logging**: To compare training costs, wrap the training loop timing. Add `time.time()` before and after `train_one_epoch` in `train.py` (or measure externally). The overhead of the KSD correction should be small relative to total epoch time.
1. **Seeding**: Use different seeds for training (seed=0,1,2) and evaluation (eval_seed in evaluate_all.py). Training seeds control the SDE noise and initialization. Evaluation seeds control which samples are generated for metrics.
1. **Failing gracefully**: If a checkpoint is missing (training crashed or was skipped), `evaluate_all.py` skips it and continues. The results tables will show gaps.
1. **The RotGMM energy must handle device transfer**: The rotation matrix and centers are torch tensors that need to be on the same device as the input. The `_to_device` method handles this. Make sure it’s called before `eval()`.