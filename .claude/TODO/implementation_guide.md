# Implementation Guide: KSD-Augmented ASBS (Stein Variational Adjoint Sampler)

## Complete Instructions for Modification, Training, Evaluation, and Results

-----

## 0. Strategy

We modify the `adjoint_samplers` codebase **minimally** by adding new files that **inherit** from existing classes. The only existing file we touch is adding new config entries. The baseline is always reproducible by running the original configs.

### File Plan

```
adjoint_samplers/                     # EXISTING REPO
├── adjoint_samplers/
│   ├── components/
│   │   ├── matcher.py                # EXISTING — do not modify
│   │   ├── ksd_matcher.py            # NEW — KSD-augmented matchers
│   │   ├── stein_kernel.py           # NEW — Stein kernel computations
│   │   ├── evaluator.py              # EXISTING — do not modify
│   │   └── ...
│   ├── energies/                     # EXISTING — do not modify
│   └── utils/                        # EXISTING — do not modify
├── configs/
│   ├── experiment/
│   │   ├── dw4_asbs.yaml             # EXISTING baseline
│   │   ├── dw4_ksd_asbs.yaml         # NEW — KSD-augmented DW4
│   │   ├── lj13_ksd_asbs.yaml        # NEW — KSD-augmented LJ13
│   │   └── lj55_ksd_asbs.yaml        # NEW — KSD-augmented LJ55
│   └── matcher/
│       ├── adjoint_ve.yaml           # EXISTING
│       └── ksd_adjoint_ve.yaml       # NEW
├── train.py                          # EXISTING — do not modify
├── evaluate_comparison.py            # NEW — head-to-head comparison script
├── generate_results.py               # NEW — generates RESULTS.md with figures
└── docs/
    ├── STEIN_VARIATIONAL_ASBS.md     # Math spec (already written)
    └── RESULTS.md                    # NEW — auto-generated results report
```

-----

## Task 1: Stein Kernel Module (`adjoint_samplers/components/stein_kernel.py`)

This module computes the Stein kernel, KSD, and the Stein kernel gradient needed for the augmented adjoint.

```python
"""
adjoint_samplers/components/stein_kernel.py

Stein kernel computations for KSD-augmented ASBS.

All functions operate on batches of terminal samples and precomputed scores.
No dependency on the energy function — only on scores s_p(x) = -grad_E(x).

See docs/STEIN_VARIATIONAL_ASBS.md Sections 1, 5 for mathematical details.
"""

import torch
from typing import Optional


@torch.no_grad()
def median_bandwidth(samples: torch.Tensor) -> torch.Tensor:
    """Compute median heuristic bandwidth for RBF kernel.

    Args:
        samples: (N, D) tensor of sample positions

    Returns:
        ell: scalar tensor, the median pairwise distance
    """
    N = samples.shape[0]

    # For large N, subsample to avoid O(N^2) memory
    if N > 5000:
        idx = torch.randperm(N, device=samples.device)[:5000]
        samples = samples[idx]
        N = 5000

    # Pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    sq_norms = (samples ** 2).sum(dim=1)  # (N,)
    sq_dists = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * samples @ samples.T  # (N, N)
    sq_dists = sq_dists.clamp(min=0)  # Numerical safety

    # Extract upper triangle (no diagonal)
    mask = torch.triu(torch.ones(N, N, device=samples.device, dtype=torch.bool), diagonal=1)
    pairwise_dists = sq_dists[mask].sqrt()

    return pairwise_dists.median()


@torch.no_grad()
def compute_stein_kernel_matrix(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute Stein kernel matrix K_p where (K_p)_{ij} = k_p(x_i, x_j).

    k_p(x, x') = K * [s^T s' + s^T δ/ℓ² − s'^T δ/ℓ² + d/ℓ² − r²/ℓ⁴]

    where K = exp(-r²/(2ℓ²)), δ = x - x', r² = ||δ||², s = s_p(x), s' = s_p(x').
    See STEIN_VARIATIONAL_ASBS.md Proposition 1.8.

    Args:
        samples: (N, D) terminal sample positions
        scores: (N, D) scores s_p(x_i) = -grad_E(x_i)
        ell: scalar bandwidth

    Returns:
        K_p: (N, N) Stein kernel matrix
    """
    N, D = samples.shape
    ell2 = ell ** 2
    ell4 = ell ** 4

    # Pairwise differences: δ[i,j] = x_i - x_j
    # Don't materialize (N, N, D) — compute terms directly

    # Squared distances: r²[i,j] = ||x_i - x_j||²
    sq_norms = (samples ** 2).sum(dim=1)  # (N,)
    sq_dists = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * samples @ samples.T
    sq_dists = sq_dists.clamp(min=0)

    # RBF kernel: K[i,j] = exp(-r²/(2ℓ²))
    K = torch.exp(-sq_dists / (2 * ell2))

    # Term 1: s_i^T s_j
    T1 = scores @ scores.T  # (N, N)

    # Term 2: s_i^T δ_{ij} / ℓ² = s_i^T (x_i - x_j) / ℓ²
    # = (s_i^T x_i - s_i^T x_j) / ℓ²
    sx = (scores * samples).sum(dim=1)  # (N,) : s_i^T x_i
    T2 = (sx.unsqueeze(1) - scores @ samples.T) / ell2  # (N, N)

    # Term 3: -s_j^T δ_{ij} / ℓ² = -(s_j^T x_i - s_j^T x_j) / ℓ²
    # = -(x_i^T s_j - s_j^T x_j) / ℓ²
    T3 = -(samples @ scores.T - sx.unsqueeze(0)) / ell2  # (N, N)

    # Term 4: d/ℓ² - r²/ℓ⁴
    T4 = D / ell2 - sq_dists / ell4

    # Stein kernel: K_p = K * (T1 + T2 + T3 + T4)
    Gamma = T1 + T2 + T3 + T4
    K_p = K * Gamma

    return K_p


@torch.no_grad()
def compute_ksd_squared(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute V-statistic estimate of KSD².

    Args:
        samples: (N, D)
        scores: (N, D) s_p(x_i) = -grad_E(x_i)
        ell: bandwidth (None = median heuristic)

    Returns:
        ksd_sq: scalar tensor
    """
    if ell is None:
        ell = median_bandwidth(samples)

    K_p = compute_stein_kernel_matrix(samples, scores, ell)
    N = samples.shape[0]
    return K_p.sum() / (N * N)


@torch.no_grad()
def compute_stein_kernel_gradient(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of Stein kernel gradients for each particle.

    For particle i, computes:
        Δ_i = Σ_j ∇_x k_p(x_i, x_j)

    using the DETACHED gradient (Hessian-free approximation):
        ∇_x k_p(x,x') ≈ K * [-δ/ℓ² · Γ + (s-s')/ℓ² - 2δ/ℓ⁴]

    See STEIN_VARIATIONAL_ASBS.md Section 5.5, 6.2.

    Args:
        samples: (N, D) terminal positions
        scores: (N, D) scores s_p(x_i)
        ell: scalar bandwidth

    Returns:
        grad_sum: (N, D) where grad_sum[i] = Σ_j ∇_x k_p(x_i, x_j)
    """
    N, D = samples.shape
    ell2 = ell ** 2
    ell4 = ell ** 4

    # Pairwise: δ[i,j,d] = x_i^d - x_j^d, shape (N, N, D)
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dists = (delta ** 2).sum(dim=-1)  # (N, N)

    # RBF kernel
    K = torch.exp(-sq_dists / (2 * ell2))  # (N, N)

    # Gamma[i,j] scalar factor
    # s_i^T s_j
    ss = scores @ scores.T  # (N, N)
    # s_i^T δ_{ij} / ℓ²
    s_dot_delta = torch.einsum('id,ijd->ij', scores, delta) / ell2  # (N, N)
    # s_j^T δ_{ij} / ℓ²
    sp_dot_delta = torch.einsum('jd,ijd->ij', scores, delta) / ell2  # (N, N)

    Gamma = ss + s_dot_delta - sp_dot_delta + D / ell2 - sq_dists / ell4  # (N, N)

    # Detached gradient (no Hessian terms):
    # ∇_x k_p(x_i, x_j) ≈ K_{ij} * [-δ_{ij}/ℓ² · Γ_{ij} + (s_i - s_j)/ℓ² - 2δ_{ij}/ℓ⁴]

    # Part A: -δ/ℓ² * Γ  → shape (N, N, D)
    partA = -delta / ell2 * Gamma.unsqueeze(-1)  # (N, N, D)

    # Part B: (s_i - s_j)/ℓ²  → shape (N, N, D)
    partB = (scores.unsqueeze(1) - scores.unsqueeze(0)) / ell2  # (N, N, D)

    # Part C: -2δ/ℓ⁴  → shape (N, N, D)
    partC = -2 * delta / ell4  # (N, N, D)

    # Full gradient: K * (A + B + C)
    grad_kp = K.unsqueeze(-1) * (partA + partB + partC)  # (N, N, D)

    # Sum over j for each i
    grad_sum = grad_kp.sum(dim=1)  # (N, D)

    return grad_sum


# =============================================================================
# Memory-efficient version for large N (avoids N×N×D tensors)
# =============================================================================

@torch.no_grad()
def compute_stein_kernel_gradient_efficient(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Memory-efficient version of compute_stein_kernel_gradient.

    Processes pairs in chunks to avoid materializing (N, N, D) tensors.

    Args:
        samples: (N, D)
        scores: (N, D)
        ell: scalar bandwidth
        chunk_size: number of j particles per chunk

    Returns:
        grad_sum: (N, D)
    """
    N, D = samples.shape
    ell2 = ell ** 2
    ell4 = ell ** 4
    grad_sum = torch.zeros_like(samples)

    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)
        M = j_end - j_start

        x_j = samples[j_start:j_end]  # (M, D)
        s_j = scores[j_start:j_end]   # (M, D)

        # δ[i, m, d] = x_i - x_j_m
        delta = samples.unsqueeze(1) - x_j.unsqueeze(0)  # (N, M, D)
        sq_dists = (delta ** 2).sum(dim=-1)  # (N, M)

        K = torch.exp(-sq_dists / (2 * ell2))  # (N, M)

        # Gamma
        ss = scores @ s_j.T  # (N, M)
        s_dot_d = torch.einsum('id,imd->im', scores, delta) / ell2
        sp_dot_d = torch.einsum('md,imd->im', s_j, delta) / ell2

        Gamma = ss + s_dot_d - sp_dot_d + D / ell2 - sq_dists / ell4

        # Gradient components
        partA = -delta / ell2 * Gamma.unsqueeze(-1)
        partB = (scores.unsqueeze(1) - s_j.unsqueeze(0)) / ell2
        partC = -2 * delta / ell4

        grad_chunk = K.unsqueeze(-1) * (partA + partB + partC)  # (N, M, D)
        grad_sum += grad_chunk.sum(dim=1)  # (N, D)

    return grad_sum
```

### 1.1 Unit Test for Stein Kernel

Add this test at the bottom of the file or as a separate test script:

```python
def _test_stein_kernel():
    """Test: KSD should be ~0 for true samples, >0 for biased samples."""
    torch.manual_seed(42)
    d = 8
    N = 500

    # True samples from N(0, I)
    x_true = torch.randn(N, d)
    s_true = -x_true  # score of N(0,I) is -x

    ksd_true = compute_ksd_squared(x_true, s_true)
    print(f"KSD^2 (true samples from N(0,I)): {ksd_true.item():.6f}")

    # Biased samples: shifted
    x_biased = torch.randn(N, d) + 1.0
    s_biased = -x_biased  # score of N(0,I) evaluated at shifted locations
    ksd_biased = compute_ksd_squared(x_biased, s_biased)
    print(f"KSD^2 (biased samples, shift=1): {ksd_biased.item():.6f}")

    assert ksd_biased > ksd_true * 5, "KSD should be much larger for biased samples"

    # Test gradient shape
    ell = median_bandwidth(x_true)
    grad = compute_stein_kernel_gradient(x_true, s_true, ell)
    assert grad.shape == (N, d), f"Expected ({N}, {d}), got {grad.shape}"

    # Test efficient version matches
    grad_eff = compute_stein_kernel_gradient_efficient(x_true, s_true, ell, chunk_size=128)
    diff = (grad - grad_eff).abs().max().item()
    print(f"Max diff (standard vs efficient): {diff:.2e}")
    assert diff < 1e-4, f"Efficient version diverges: {diff}"

    print("All Stein kernel tests passed!")


if __name__ == "__main__":
    _test_stein_kernel()
```

-----

## Task 2: KSD-Augmented Matcher (`adjoint_samplers/components/ksd_matcher.py`)

This module creates new matcher classes that inherit from the existing VE and VP matchers, overriding only `populate_buffer` to add the KSD correction to the adjoint.

```python
"""
adjoint_samplers/components/ksd_matcher.py

KSD-augmented adjoint matchers. Inherit from existing matchers and override
only the populate_buffer method to add the inter-particle Stein kernel
gradient correction to the adjoint terminal condition.

See docs/STEIN_VARIATIONAL_ASBS.md Section 4, 6 for the math.
"""

import torch
from adjoint_samplers.components.sde import sdeint
from adjoint_samplers.components.matcher import (
    AdjointVEMatcher,
    AdjointVPMatcher,
)
from adjoint_samplers.components.stein_kernel import (
    median_bandwidth,
    compute_stein_kernel_gradient,
    compute_stein_kernel_gradient_efficient,
    compute_ksd_squared,
)


class KSDAdjointVEMatcher(AdjointVEMatcher):
    """AdjointVEMatcher with KSD correction on the adjoint terminal condition.

    For VE-SDE (no drift), the adjoint is constant in time:
        Y_t^i = Y_1^i = -(1/N) * ∇Φ₀(x₁ⁱ) - (λ/N²) * Σⱼ ∇ₓ kₚ(x₁ⁱ, x₁ʲ)

    The first term is the standard ASBS adjoint.
    The second term is the KSD correction that prevents mode collapse.
    """
    def __init__(
        self,
        ksd_lambda: float = 1.0,
        ksd_bandwidth: float | None = None,
        ksd_max_particles: int = 2048,
        ksd_efficient_threshold: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ksd_lambda = ksd_lambda
        self.ksd_bandwidth = ksd_bandwidth
        self.ksd_max_particles = ksd_max_particles
        self.ksd_efficient_threshold = ksd_efficient_threshold

        # Logging
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        """Override to add KSD correction to the adjoint."""

        # Step 1: Standard forward simulation (unchanged)
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )

        # Step 2: Standard adjoint computation (unchanged)
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        # Step 3: KSD correction (NEW)
        if self.ksd_lambda > 0 and not is_asbs_init_stage:
            adjoint1 = self._apply_ksd_correction(x1, adjoint1)

        # Step 4: Store in buffer (unchanged)
        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })

    @torch.no_grad()
    def _apply_ksd_correction(
        self,
        x1: torch.Tensor,
        adjoint1: torch.Tensor,
    ) -> torch.Tensor:
        """Add the KSD gradient correction to the adjoint.

        adjoint1_new[i] = adjoint1[i] + (λ/N²) * Σⱼ ∇ₓ kₚ(x₁ⁱ, x₁ʲ)

        Note: adjoint1 from the matcher is ∇Φ₀(x₁) (forces, positive).
        The AM target is -adjoint1. The KSD correction gradient points
        in the direction that DECREASES KSD, so we ADD it to adjoint1
        (which represents the "forces" = ∇Φ₀, i.e., the gradient of
        the terminal cost that the controller should cancel).

        See STEIN_VARIATIONAL_ASBS.md Section 4.1:
            Y₁ⁱ = -(1/N)∇Φ₀ - (λ/N²)Σⱼ ∇ₓkₚ
        Since adjoint1 = ∇Φ₀ and the matcher stores adjoint1 (then negates
        in prepare_target), we need:
            adjoint1_corrected = adjoint1 + (λ/N) * (1/N)Σⱼ ∇ₓkₚ
        """
        N, D = x1.shape
        device = x1.device

        # Subsample if too many particles
        if N > self.ksd_max_particles:
            idx = torch.randperm(N, device=device)[:self.ksd_max_particles]
            x1_sub = x1[idx]
        else:
            x1_sub = x1

        N_sub = x1_sub.shape[0]

        # Compute scores at terminal samples
        # The energy's score method returns s_p(x) = -grad_E(x)
        # We need to call the energy directly since it's in grad_term_cost
        with torch.enable_grad():
            x1_req = x1_sub.clone().detach().requires_grad_(True)
            E = self.grad_term_cost.energy.eval(x1_req)
            scores = -torch.autograd.grad(E.sum(), x1_req)[0]  # s_p = -∇E

        # Bandwidth
        if self.ksd_bandwidth is not None:
            ell = torch.tensor(self.ksd_bandwidth, device=device)
        else:
            ell = median_bandwidth(x1_sub)

        # Compute Stein kernel gradient sum
        if N_sub > self.ksd_efficient_threshold:
            grad_sum = compute_stein_kernel_gradient_efficient(
                x1_sub, scores, ell, chunk_size=256
            )
        else:
            grad_sum = compute_stein_kernel_gradient(x1_sub, scores, ell)

        # KSD correction: (λ / N²) * Σⱼ ∇ₓkₚ(xᵢ, xⱼ)
        ksd_correction = (self.ksd_lambda / (N_sub ** 2)) * grad_sum

        # Logging
        self._last_ksd = compute_ksd_squared(x1_sub, scores, ell).item()
        self._last_ksd_grad_norm = ksd_correction.norm(dim=-1).mean().item()

        # If we subsampled, we only have corrections for the subset.
        # For the full batch, apply correction only to the subset indices.
        if N > self.ksd_max_particles:
            full_correction = torch.zeros_like(adjoint1)
            full_correction[idx] = ksd_correction
            adjoint1 = adjoint1 + full_correction
        else:
            adjoint1 = adjoint1 + ksd_correction

        return adjoint1


class KSDAdjointVPMatcher(AdjointVPMatcher):
    """KSD-augmented matcher for VP-SDE. Same logic, different base class."""
    def __init__(
        self,
        ksd_lambda: float = 1.0,
        ksd_bandwidth: float | None = None,
        ksd_max_particles: int = 2048,
        ksd_efficient_threshold: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ksd_lambda = ksd_lambda
        self.ksd_bandwidth = ksd_bandwidth
        self.ksd_max_particles = ksd_max_particles
        self.ksd_efficient_threshold = ksd_efficient_threshold
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        if self.ksd_lambda > 0 and not is_asbs_init_stage:
            # Reuse the same _apply_ksd_correction logic
            adjoint1 = KSDAdjointVEMatcher._apply_ksd_correction(self, x1, adjoint1)

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })
```

-----

## Task 3: Config Files

### 3.1 New Matcher Config (`configs/matcher/ksd_adjoint_ve.yaml`)

```yaml
_target_: adjoint_samplers.components.ksd_matcher.KSDAdjointVEMatcher

ksd_lambda: ${ksd_lambda}
ksd_bandwidth: null  # null = median heuristic
ksd_max_particles: 2048
ksd_efficient_threshold: 1024

grad_state_cost:
  _target_: adjoint_samplers.components.state_cost.ZeroGradStateCost

buffer:
  _target_: adjoint_samplers.components.buffer.BatchBuffer
  buffer_size: ${adjoint_matcher.buffer_size}
```

### 3.2 DW4 KSD-ASBS Config (`configs/experiment/dw4_ksd_asbs.yaml`)

```yaml
# @package _global_

defaults:
  - /problem: dw4
  - /source: harmonic
  - /sde@ref_sde: graph_ve
  - /model@controller: egnn
  - /state_cost: zero
  - /term_cost: graph_corrector_term_cost
  - /matcher@adjoint_matcher: ksd_adjoint_ve   # <-- ONLY CHANGE: use KSD matcher
  - /model@corrector: egnn
  - /matcher@corrector_matcher: corrector

exp_name: dw4_ksd_asbs

# === source ===
scale: 2

# === sde ===
nfe: 200
sigma_max: 1
sigma_min: 0.001
rescale_t: null

# === KSD parameters (NEW) ===
ksd_lambda: 1.0   # Start with 1.0, ablate over {0.1, 0.5, 1.0, 5.0, 10.0}

# === training ===
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

# === eval & logging ===
use_wandb: true
```

### 3.3 LJ13 KSD-ASBS Config (`configs/experiment/lj13_ksd_asbs.yaml`)

Same as `lj13_asbs.yaml` but replace the matcher line:

```yaml
# @package _global_

defaults:
  - /problem: lj13
  - /source: harmonic
  - /sde@ref_sde: graph_ve
  - /model@controller: egnn
  - /state_cost: zero
  - /term_cost: graph_corrector_term_cost
  - /matcher@adjoint_matcher: ksd_adjoint_ve   # <-- KSD matcher
  - /model@corrector: egnn
  - /matcher@corrector_matcher: corrector

exp_name: lj13_ksd_asbs

# Copy all other settings from lj13_asbs.yaml exactly
# Add:
ksd_lambda: 1.0
```

### 3.4 LJ55 KSD-ASBS Config (`configs/experiment/lj55_ksd_asbs.yaml`)

Same pattern. Copy `lj55_asbs.yaml`, change matcher to `ksd_adjoint_ve`, add `ksd_lambda: 1.0`.

-----

## Task 4: Logging KSD During Training

To monitor the KSD correction, add logging to `train.py`. This is the **one small modification** to an existing file.

In `train.py`, after line 163 (`writer.log({...}, step=epoch)`), add:

```python
# Log KSD diagnostics if using KSD matcher
if hasattr(matcher, '_last_ksd'):
    writer.log({
        f"{stage}_ksd_squared": matcher._last_ksd,
        f"{stage}_ksd_grad_norm": matcher._last_ksd_grad_norm,
    }, step=epoch)
```

If you prefer not to modify `train.py` at all, you can skip this — the evaluation script will compute KSD independently. But having it during training is useful for debugging.

-----

## Task 5: Evaluation and Comparison Script (`evaluate_comparison.py`)

This script loads trained checkpoints from both baseline ASBS and KSD-augmented ASBS, runs evaluation, and produces a structured comparison.

```python
"""
evaluate_comparison.py

Head-to-head comparison between vanilla ASBS and KSD-augmented ASBS.
Loads both checkpoints, generates samples, computes metrics, outputs results.

Usage:
    python evaluate_comparison.py \
        --experiment dw4 \
        --baseline_ckpt results/dw4_asbs/checkpoints/checkpoint_latest.pt \
        --ksd_ckpt results/dw4_ksd_asbs/checkpoints/checkpoint_latest.pt \
        --output_dir comparison_results/dw4 \
        --n_seeds 10 \
        --n_samples 2000
"""

import argparse
import json
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.evaluator import SyntheticEenergyEvaluator
from adjoint_samplers.components.stein_kernel import (
    compute_ksd_squared,
    median_bandwidth,
)
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
import adjoint_samplers.utils.train_utils as train_utils

import ot as pot


def load_model(cfg_path, ckpt_path, device):
    """Load a trained ASBS model from config + checkpoint."""
    cfg = OmegaConf.load(cfg_path)

    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt["controller"])

    timesteps_cfg = {
        't0': cfg.timesteps.t0,
        't1': cfg.timesteps.t1,
        'steps': cfg.timesteps.steps,
        'rescale_t': cfg.timesteps.rescale_t,
    }

    return sde, source, energy, timesteps_cfg


@torch.no_grad()
def generate_samples(sde, source, timesteps, n_samples, batch_size, device):
    """Generate terminal samples."""
    x1_list = []
    n_gen = 0
    while n_gen < n_samples:
        b = min(batch_size, n_samples - n_gen)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**timesteps).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1)
        n_gen += b
    return torch.cat(x1_list, dim=0)[:n_samples]


def compute_metrics(samples, ref_samples, energy, n_particles, spatial_dim):
    """Compute all evaluation metrics."""
    B = min(len(samples), len(ref_samples))
    idx_gen = torch.randperm(len(samples))[:B]
    idx_ref = torch.randperm(len(ref_samples))[:B]
    gen = samples[idx_gen]
    ref = ref_samples[idx_ref]

    metrics = {}

    # Energies
    gen_E = energy.eval(gen)
    ref_E = energy.eval(ref)
    metrics['mean_energy_gen'] = gen_E.mean().item()
    metrics['mean_energy_ref'] = ref_E.mean().item()
    metrics['energy_w2'] = float(pot.emd2_1d(
        ref_E.cpu().numpy(), gen_E.cpu().numpy()
    ) ** 0.5)

    # Interatomic distances
    gen_dist = interatomic_dist(gen, n_particles, spatial_dim)
    ref_dist = interatomic_dist(ref, n_particles, spatial_dim)
    metrics['dist_w2'] = float(pot.emd2_1d(
        gen_dist.cpu().numpy().reshape(-1),
        ref_dist.cpu().numpy().reshape(-1),
    ))

    # Particle W2
    M = dist_point_clouds(
        gen.reshape(-1, n_particles, spatial_dim).cpu(),
        ref.reshape(-1, n_particles, spatial_dim).cpu(),
    )
    a = torch.ones(M.shape[0]) / M.shape[0]
    b = torch.ones(M.shape[0]) / M.shape[0]
    metrics['eq_w2'] = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)

    # KSD
    scores = energy.score(gen)
    ell = median_bandwidth(gen)
    metrics['ksd_squared'] = compute_ksd_squared(gen, scores, ell).item()

    return metrics


def run_comparison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline from {args.baseline_ckpt}...")
    sde_base, source_base, energy, ts_base = load_model(
        args.baseline_cfg, args.baseline_ckpt, device
    )

    print(f"Loading KSD-augmented from {args.ksd_ckpt}...")
    sde_ksd, source_ksd, _, ts_ksd = load_model(
        args.ksd_cfg, args.ksd_ckpt, device
    )

    # Load reference samples
    ref_path = f"data/test_split_{args.experiment.upper()}.npy"
    if args.experiment.lower().startswith('lj'):
        ref_path = f"data/test_split_{args.experiment.upper()}-1000.npy"
        if args.experiment.lower() == 'lj55':
            ref_path = f"data/test_split_LJ55-1000-part1.npy"
    ref_samples = torch.tensor(np.load(ref_path, allow_pickle=True)).to(device)

    n_particles = energy.n_particles
    spatial_dim = energy.n_spatial_dim

    print(f"\nRunning {args.n_seeds} seeds, {args.n_samples} samples each...")

    all_baseline = []
    all_ksd = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed * 1000)
        np.random.seed(seed * 1000)

        # Generate baseline samples
        samples_base = generate_samples(
            sde_base, source_base, ts_base,
            args.n_samples, args.batch_size, device
        )
        m_base = compute_metrics(
            samples_base, ref_samples, energy, n_particles, spatial_dim
        )
        all_baseline.append(m_base)

        # Generate KSD samples
        samples_ksd = generate_samples(
            sde_ksd, source_ksd, ts_ksd,
            args.n_samples, args.batch_size, device
        )
        m_ksd = compute_metrics(
            samples_ksd, ref_samples, energy, n_particles, spatial_dim
        )
        all_ksd.append(m_ksd)

        print(f"  seed {seed}: baseline energy_w2={m_base['energy_w2']:.4f}, "
              f"ksd_asbs energy_w2={m_ksd['energy_w2']:.4f}")

    # Aggregate
    results = {'experiment': args.experiment, 'n_seeds': args.n_seeds,
               'n_samples': args.n_samples}

    for key in all_baseline[0].keys():
        base_vals = [m[key] for m in all_baseline]
        ksd_vals = [m[key] for m in all_ksd]
        results[f'baseline_{key}_mean'] = float(np.mean(base_vals))
        results[f'baseline_{key}_std'] = float(np.std(base_vals))
        results[f'ksd_{key}_mean'] = float(np.mean(ksd_vals))
        results[f'ksd_{key}_std'] = float(np.std(ksd_vals))

    # Save
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'comparison.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['dw4', 'lj13', 'lj55'])
    parser.add_argument('--baseline_cfg', type=str, required=True)
    parser.add_argument('--baseline_ckpt', type=str, required=True)
    parser.add_argument('--ksd_cfg', type=str, required=True)
    parser.add_argument('--ksd_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=2000)
    args = parser.parse_args()
    run_comparison(args)
```

-----

## Task 6: Results Generation Script (`generate_results.py`)

This script reads comparison JSON files and generates `RESULTS.md` with tables and figures.

```python
"""
generate_results.py

Reads comparison results and generates RESULTS.md with:
- Summary tables comparing baseline vs KSD-augmented ASBS
- Figures: bar charts, training curves, sample visualizations
- Ablation results over ksd_lambda

Usage:
    python generate_results.py \
        --results_dir comparison_results \
        --output docs/RESULTS.md
"""

import argparse
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EXPERIMENTS = ['dw4', 'lj13', 'lj55']
METRICS = ['energy_w2', 'dist_w2', 'eq_w2', 'ksd_squared', 'mean_energy_gen']
METRIC_LABELS = {
    'energy_w2': 'Energy W₂ ↓',
    'dist_w2': 'Interatomic Dist W₂ ↓',
    'eq_w2': 'Particle Config W₂ ↓',
    'ksd_squared': 'KSD² ↓',
    'mean_energy_gen': 'Mean Energy',
}


def load_results(results_dir):
    """Load all comparison JSONs."""
    data = {}
    for exp in EXPERIMENTS:
        path = Path(results_dir) / exp / 'comparison.json'
        if path.exists():
            with open(path) as f:
                data[exp] = json.load(f)
    return data


def make_comparison_table(data, experiment):
    """Generate a markdown table for one experiment."""
    r = data[experiment]
    lines = []
    lines.append(f"### {experiment.upper()}")
    lines.append("")
    lines.append("| Metric | Baseline ASBS | KSD-Augmented ASBS | Δ (%) |")
    lines.append("|---|---|---|---|")

    for metric in METRICS:
        bm = r.get(f'baseline_{metric}_mean', float('nan'))
        bs = r.get(f'baseline_{metric}_std', float('nan'))
        km = r.get(f'ksd_{metric}_mean', float('nan'))
        ks = r.get(f'ksd_{metric}_std', float('nan'))

        if bm != 0 and not np.isnan(bm):
            delta = (km - bm) / abs(bm) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "—"

        label = METRIC_LABELS.get(metric, metric)
        lines.append(
            f"| {label} | {bm:.4f} ± {bs:.4f} | {km:.4f} ± {ks:.4f} | {delta_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def make_bar_chart(data, results_dir):
    """Create a grouped bar chart comparing metrics across experiments."""
    fig_dir = Path(results_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ['energy_w2', 'dist_w2', 'eq_w2']
    experiments = [e for e in EXPERIMENTS if e in data]

    if not experiments:
        return []

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    paths = []
    for ax, metric in zip(axes, metrics_to_plot):
        base_means = [data[e].get(f'baseline_{metric}_mean', 0) for e in experiments]
        base_stds = [data[e].get(f'baseline_{metric}_std', 0) for e in experiments]
        ksd_means = [data[e].get(f'ksd_{metric}_mean', 0) for e in experiments]
        ksd_stds = [data[e].get(f'ksd_{metric}_std', 0) for e in experiments]

        x = np.arange(len(experiments))
        w = 0.35

        ax.bar(x - w/2, base_means, w, yerr=base_stds, label='Baseline ASBS',
               color='#1f77b4', capsize=3, edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, ksd_means, w, yerr=ksd_stds, label='KSD-ASBS',
               color='#ff7f0e', capsize=3, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([e.upper() for e in experiments])
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Baseline ASBS vs KSD-Augmented ASBS', fontsize=14)
    fig.tight_layout()

    path = fig_dir / 'comparison_bars.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # KSD comparison
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    base_ksd = [data[e].get('baseline_ksd_squared_mean', 0) for e in experiments]
    ksd_ksd = [data[e].get('ksd_ksd_squared_mean', 0) for e in experiments]
    x = np.arange(len(experiments))
    ax2.bar(x - w/2, base_ksd, w, label='Baseline', color='#1f77b4',
            edgecolor='black', linewidth=0.5)
    ax2.bar(x + w/2, ksd_ksd, w, label='KSD-ASBS', color='#ff7f0e',
            edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.upper() for e in experiments])
    ax2.set_ylabel('KSD²')
    ax2.set_title('Kernel Stein Discrepancy (lower = better)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()

    path2 = fig_dir / 'ksd_comparison.png'
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    paths.append(path2)

    return paths


def generate_results_md(data, results_dir, output_path):
    """Generate the full RESULTS.md."""
    fig_paths = make_bar_chart(data, results_dir)

    lines = []
    lines.append("# Results: KSD-Augmented ASBS vs Baseline ASBS")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("This document compares the **baseline ASBS** (Adjoint Schrödinger Bridge Sampler)")
    lines.append("against **KSD-Augmented ASBS**, which adds an inter-particle Stein kernel gradient")
    lines.append("correction to the adjoint terminal condition to prevent mode collapse.")
    lines.append("")
    lines.append("The KSD correction modifies the adjoint as:")
    lines.append("")
    lines.append("$$Y_1^i = -\\frac{1}{N}\\nabla\\Phi_0(X_1^i) - \\frac{\\lambda}{N^2}\\sum_j \\nabla_x k_p(X_1^i, X_1^j)$$")
    lines.append("")
    lines.append("See `docs/STEIN_VARIATIONAL_ASBS.md` for the full mathematical derivation.")
    lines.append("")

    # Figures
    if fig_paths:
        lines.append("## Comparison Figures")
        lines.append("")
        for p in fig_paths:
            rel = p.relative_to(Path(results_dir).parent) if Path(results_dir).parent != Path('.') else p
            lines.append(f"![{p.stem}]({rel})")
            lines.append("")

    # Per-experiment tables
    lines.append("## Detailed Results")
    lines.append("")
    for exp in EXPERIMENTS:
        if exp in data:
            lines.append(make_comparison_table(data, exp))

    # Overall verdict
    lines.append("## Verdict")
    lines.append("")

    improved = 0
    total = 0
    for exp in EXPERIMENTS:
        if exp in data:
            for metric in ['energy_w2', 'dist_w2', 'eq_w2']:
                bm = data[exp].get(f'baseline_{metric}_mean', float('inf'))
                km = data[exp].get(f'ksd_{metric}_mean', float('inf'))
                if km < bm:
                    improved += 1
                total += 1

    if total > 0:
        pct = improved / total * 100
        lines.append(f"KSD-Augmented ASBS improved on **{improved}/{total}** ({pct:.0f}%) of metric-experiment combinations.")
    lines.append("")

    for exp in EXPERIMENTS:
        if exp in data:
            bm = data[exp].get('baseline_ksd_squared_mean', float('inf'))
            km = data[exp].get('ksd_ksd_squared_mean', float('inf'))
            if km < bm:
                lines.append(f"- **{exp.upper()}**: KSD² reduced from {bm:.6f} to {km:.6f} ({(1-km/bm)*100:.1f}% reduction)")
            else:
                lines.append(f"- **{exp.upper()}**: KSD² baseline={bm:.6f}, KSD-ASBS={km:.6f} (no improvement)")

    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# Train baseline")
    lines.append("python train.py experiment=dw4_asbs seed=0,1,2 -m")
    lines.append("")
    lines.append("# Train KSD-augmented")
    lines.append("python train.py experiment=dw4_ksd_asbs seed=0,1,2 -m")
    lines.append("")
    lines.append("# Compare")
    lines.append("python evaluate_comparison.py \\")
    lines.append("    --experiment dw4 \\")
    lines.append("    --baseline_cfg outputs/dw4_asbs/0/config.yaml \\")
    lines.append("    --baseline_ckpt outputs/dw4_asbs/0/checkpoints/checkpoint_latest.pt \\")
    lines.append("    --ksd_cfg outputs/dw4_ksd_asbs/0/config.yaml \\")
    lines.append("    --ksd_ckpt outputs/dw4_ksd_asbs/0/checkpoints/checkpoint_latest.pt \\")
    lines.append("    --output_dir comparison_results/dw4")
    lines.append("")
    lines.append("# Generate this report")
    lines.append("python generate_results.py --results_dir comparison_results --output docs/RESULTS.md")
    lines.append("```")

    # Write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write("\n".join(lines))
    print(f"RESULTS.md written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='comparison_results')
    parser.add_argument('--output', type=str, default='docs/RESULTS.md')
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print("No results found! Run evaluate_comparison.py first.")
    else:
        generate_results_md(data, args.results_dir, args.output)
```

-----

## Task 7: Full Pipeline — Step by Step

### 7.1 Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/adjoint_samplers.git
cd adjoint_samplers
git checkout -b ksd-augmented-asbs

# Install
micromamba env create -f environment.yml
micromamba activate adjoint_samplers

# Download reference samples
bash scripts/download.sh
```

### 7.2 Create New Files

Create the following files exactly as specified in Tasks 1–3:

1. `adjoint_samplers/components/stein_kernel.py` (Task 1)
1. `adjoint_samplers/components/ksd_matcher.py` (Task 2)
1. `configs/matcher/ksd_adjoint_ve.yaml` (Task 3.1)
1. `configs/experiment/dw4_ksd_asbs.yaml` (Task 3.2)
1. `configs/experiment/lj13_ksd_asbs.yaml` (Task 3.3)
1. `configs/experiment/lj55_ksd_asbs.yaml` (Task 3.4)

### 7.3 Test the Stein Kernel Module

```bash
python -m adjoint_samplers.components.stein_kernel
# Should print: "All Stein kernel tests passed!"
```

### 7.4 Train Baseline (if not already done)

```bash
# DW4 baseline — 3 seeds
python train.py experiment=dw4_asbs seed=0,1,2 -m use_wandb=false

# LJ13 baseline — 3 seeds
python train.py experiment=lj13_asbs seed=0,1,2 -m use_wandb=false
```

### 7.5 Train KSD-Augmented ASBS

```bash
# DW4 with KSD — 3 seeds, λ=1.0
python train.py experiment=dw4_ksd_asbs seed=0,1,2 -m use_wandb=false

# λ ablation on DW4
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=0.1 use_wandb=false
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=0.5 use_wandb=false
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=1.0 use_wandb=false
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=5.0 use_wandb=false
python train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=10.0 use_wandb=false

# LJ13 with KSD — 3 seeds
python train.py experiment=lj13_ksd_asbs seed=0,1,2 -m use_wandb=false
```

### 7.6 Run Head-to-Head Comparison

```bash
# DW4
python evaluate_comparison.py \
    --experiment dw4 \
    --baseline_cfg outputs/dw4_asbs/0/config.yaml \
    --baseline_ckpt outputs/dw4_asbs/0/checkpoints/checkpoint_latest.pt \
    --ksd_cfg outputs/dw4_ksd_asbs/0/config.yaml \
    --ksd_ckpt outputs/dw4_ksd_asbs/0/checkpoints/checkpoint_latest.pt \
    --output_dir comparison_results/dw4

# LJ13
python evaluate_comparison.py \
    --experiment lj13 \
    --baseline_cfg outputs/lj13_asbs/0/config.yaml \
    --baseline_ckpt outputs/lj13_asbs/0/checkpoints/checkpoint_latest.pt \
    --ksd_cfg outputs/lj13_ksd_asbs/0/config.yaml \
    --ksd_ckpt outputs/lj13_ksd_asbs/0/checkpoints/checkpoint_latest.pt \
    --output_dir comparison_results/lj13
```

### 7.7 Generate Results Report

```bash
python generate_results.py \
    --results_dir comparison_results \
    --output docs/RESULTS.md
```

This creates `docs/RESULTS.md` with tables and figures.

-----

## Task 8: Order of Implementation

Build and test in this exact order:

1. **`stein_kernel.py`** — Run unit test. Must pass before anything else.
1. **`ksd_matcher.py`** — Imports from stein_kernel. No test yet (needs training).
1. **`ksd_adjoint_ve.yaml`** — Config for the new matcher.
1. **`dw4_ksd_asbs.yaml`** — Experiment config.
1. **Train DW4 KSD-ASBS** with `ksd_lambda=1.0` — Verify it trains without crashing. Compare loss curves to baseline. If the loss is much worse, reduce `ksd_lambda`.
1. **`evaluate_comparison.py`** — Run comparison.
1. **`generate_results.py`** — Generate report.
1. **Scale to LJ13** — Repeat steps 4–7 with LJ13 configs.
1. **Lambda ablation** — Try multiple $\lambda$ values on DW4.
1. **LJ55** — Only if DW4 and LJ13 show improvement.

-----

## Task 9: What to Look For in the Results

### 9.1 Success Criteria

**Primary metric — Mode coverage:**
The KSD-augmented ASBS should produce terminal samples that cover more modes of $p$ than baseline. This manifests as:

- Lower KSD² (directly measures distributional mismatch)
- Lower energy W₂ (better match to reference energy distribution)
- Lower eq_w₂ (better match to reference particle configurations)

**Secondary metric — No degradation:**
The KSD correction should not hurt per-particle quality. Mean energy should be similar or better than baseline.

### 9.2 What the Numbers Should Look Like

|Scenario                              |What it means                                                         |
|--------------------------------------|----------------------------------------------------------------------|
|KSD² (ksd) < KSD² (baseline)          |KSD correction is working — terminal distribution is closer to $p$    |
|energy_w2 (ksd) < energy_w2 (baseline)|Overall sample quality improved                                       |
|energy_w2 similar, but KSD² lower     |Particles are better distributed but individually similar quality     |
|KSD² similar or higher                |λ too small (correction too weak) or too large (destabilizes training)|
|Training loss diverges                |λ too large — reduce by 10×                                           |

### 9.3 Expected Outcomes by Benchmark

**DW4 (8D, multimodal):** Most likely to show improvement. The double well has clear multi-modal structure — mode collapse is the primary failure mode.

**LJ13 (39D):** Moderate improvement expected. The Stein kernel with RBF still works reasonably at 39D.

**LJ55 (165D):** Least likely to help. The RBF kernel degrades at 165D — pairwise distances concentrate, and the Stein kernel becomes nearly constant. If DW4 and LJ13 show improvement but LJ55 doesn’t, this is strong evidence that the method works but needs a better kernel for high dimensions (future work: deep kernel, graph kernel matching the EGNN architecture).

### 9.4 If Results Are Negative

Document everything. Negative results are informative:

- If KSD-ASBS performs **identically** to baseline → ASBS already covers modes well; the KSD correction is zero at convergence (consistent with Section 7.4 of the math spec)
- If KSD-ASBS performs **worse** → the detached Hessian approximation may be too crude, or λ needs tuning
- If KSD-ASBS helps on DW4 but not LJ55 → kernel scaling limitation (publishable observation)

-----

## Important Notes for Claude Code

1. **Device consistency.** All tensors in `stein_kernel.py` and `ksd_matcher.py` must stay on the same device as the samples. The energy evaluations and score computations happen on CUDA.
1. **COM-free coordinates.** DW4/LJ13/LJ55 use center-of-mass-free particle coordinates. The `Graph` mixin handles this via `propagate` and `randn_like`. The Stein kernel computation operates on the raw terminal samples $X_1$ which are already COM-free — no special handling needed.
1. **Gradient detaching.** In `_apply_ksd_correction`, the KSD correction is computed with `@torch.no_grad()`. We do NOT backprop through the KSD — it modifies the adjoint target, which is a fixed regression target for the AM loss. This is by design (see MATH_SPEC Section 5.3 / 6.2).
1. **Memory.** The `(N, N, D)` tensor in `compute_stein_kernel_gradient` costs $N^2 \times D \times 4$ bytes. For $N=512, D=8$ (DW4): 8MB — fine. For $N=512, D=165$ (LJ55): 170MB — manageable. For $N > 1024$ with high $D$, use `compute_stein_kernel_gradient_efficient` with chunking.
1. **The `ksd_lambda` parameter.** Start with 1.0. If training is unstable, reduce to 0.1. The correction should be a small perturbation to the adjoint — if `ksd_grad_norm >> adjoint_norm`, λ is too large.
1. **Hydra output paths.** Hydra saves outputs to `outputs/EXPERIMENT_NAME/SEED/`. The config is saved as `outputs/.../config.yaml` and checkpoints as `outputs/.../checkpoints/checkpoint_latest.pt`. The evaluation script needs these paths.
1. **No modification of original ASBS training code.** The `ksd_matcher.py` inherits from existing matchers. The `stein_kernel.py` is a standalone module. New config files point to the new matcher class. The `train.py` and `train_loop.py` work unchanged because the new matcher has the same interface (same `populate_buffer` signature, same buffer format).