# Copyright (c) SungminLee511. KSD-Augmented Adjoint Samplers.

"""
Stein kernel computations for KSD-augmented ASBS.

Standalone module — no dependency on the rest of the codebase.
Implements:
  - median_bandwidth: adaptive RBF bandwidth via median heuristic
  - compute_ksd_squared: V-statistic estimator of KSD²
  - compute_stein_kernel_gradient: ∇ₓ kₚ(xᵢ, xⱼ) summed over j (detached, no Hessian)
  - compute_stein_kernel_gradient_efficient: chunked version for large N

Math reference: .claude/skills/math_specs.md
"""

import torch


# =============================================================================
# Bandwidth selection
# =============================================================================

def median_bandwidth(samples: torch.Tensor) -> torch.Tensor:
    """Compute the median heuristic bandwidth for the RBF kernel.

    ℓ = median(‖xᵢ - xⱼ‖ : i < j)

    Args:
        samples: (N, D) tensor of particle positions.

    Returns:
        Scalar tensor ℓ > 0 (clamped to avoid zero).
    """
    with torch.no_grad():
        # Pairwise squared distances
        sq_dists = torch.cdist(samples, samples, p=2).pow(2)
        # Upper triangle (i < j)
        N = samples.shape[0]
        mask = torch.triu(torch.ones(N, N, device=samples.device, dtype=torch.bool), diagonal=1)
        pairwise_dists = sq_dists[mask].sqrt()
        median_dist = pairwise_dists.median()
        # Clamp to avoid zero bandwidth
        return median_dist.clamp(min=1e-5)


# =============================================================================
# KSD² computation (V-statistic)
# =============================================================================

@torch.no_grad()
def compute_ksd_squared(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KSD²(q, p) using the V-statistic estimator.

    KSD²_V = (1/N²) Σᵢⱼ kₚ(xᵢ, xⱼ)

    where kₚ is the Stein kernel with RBF base kernel:
        kₚ(x, x') = K · [s^Ts' + s^Tδ/ℓ² - s'^Tδ/ℓ² + d/ℓ² - r²/ℓ⁴]

    Args:
        samples: (N, D) terminal particle positions.
        scores: (N, D) score sₚ(xᵢ) = -∇E(xᵢ) at each particle.
        ell: scalar bandwidth. If None, uses median heuristic.

    Returns:
        Scalar tensor: KSD² estimate.
    """
    N, D = samples.shape

    if ell is None:
        ell = median_bandwidth(samples)

    ell2 = ell ** 2
    ell4 = ell ** 4

    # Pairwise differences: δ[i,j] = xᵢ - xⱼ
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dists = (delta ** 2).sum(dim=-1)  # (N, N)

    # RBF kernel
    K = torch.exp(-sq_dists / (2 * ell2))  # (N, N)

    # Stein kernel scalar factor Γ[i,j]
    # sᵢ^T sⱼ
    ss = scores @ scores.T  # (N, N)
    # sᵢ^T δᵢⱼ / ℓ²
    s_dot_delta = torch.einsum('id,ijd->ij', scores, delta) / ell2  # (N, N)
    # sⱼ^T δᵢⱼ / ℓ²
    sp_dot_delta = torch.einsum('jd,ijd->ij', scores, delta) / ell2  # (N, N)

    Gamma = ss + s_dot_delta - sp_dot_delta + D / ell2 - sq_dists / ell4  # (N, N)

    # Stein kernel: kₚ = K · Γ
    kp = K * Gamma  # (N, N)

    # V-statistic: (1/N²) Σᵢⱼ kₚ(xᵢ, xⱼ)
    return kp.sum() / (N ** 2)


# =============================================================================
# Stein kernel gradient (detached, no Hessian)
# =============================================================================

@torch.no_grad()
def compute_stein_kernel_gradient(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
) -> torch.Tensor:
    """Compute Σⱼ ∇ₓ kₚ(xᵢ, xⱼ) for each particle i (detached, no Hessian).

    Detached gradient (Hessian-free approximation):
        ∇ₓ kₚ(xᵢ, xⱼ) ≈ Kᵢⱼ · [-δᵢⱼ/ℓ² · Γᵢⱼ + (sᵢ - sⱼ)/ℓ² - 2δᵢⱼ/ℓ⁴]

    Args:
        samples: (N, D) terminal particle positions.
        scores: (N, D) score sₚ(xᵢ) = -∇E(xᵢ).
        ell: scalar RBF bandwidth.

    Returns:
        grad_sum: (N, D) — Σⱼ ∇ₓ kₚ(xᵢ, xⱼ) for each i.
    """
    N, D = samples.shape
    ell2 = ell ** 2
    ell4 = ell ** 4

    # Pairwise: δ[i,j,d] = xᵢ^d - xⱼ^d
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dists = (delta ** 2).sum(dim=-1)  # (N, N)

    # RBF kernel
    K = torch.exp(-sq_dists / (2 * ell2))  # (N, N)

    # Gamma[i,j] scalar factor
    ss = scores @ scores.T  # (N, N)
    s_dot_delta = torch.einsum('id,ijd->ij', scores, delta) / ell2  # (N, N)
    sp_dot_delta = torch.einsum('jd,ijd->ij', scores, delta) / ell2  # (N, N)

    Gamma = ss + s_dot_delta - sp_dot_delta + D / ell2 - sq_dists / ell4  # (N, N)

    # Detached gradient components:
    # Part A: -δ/ℓ² · Γ  → (N, N, D)
    partA = -delta / ell2 * Gamma.unsqueeze(-1)

    # Part B: (sᵢ - sⱼ)/ℓ²  → (N, N, D)
    partB = (scores.unsqueeze(1) - scores.unsqueeze(0)) / ell2

    # Part C: -2δ/ℓ⁴  → (N, N, D)
    partC = -2 * delta / ell4

    # Full gradient: K · (A + B + C), summed over j
    grad_kp = K.unsqueeze(-1) * (partA + partB + partC)  # (N, N, D)
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

        x_j = samples[j_start:j_end]  # (M, D)
        s_j = scores[j_start:j_end]   # (M, D)

        # δ[i, m, d] = xᵢ - xⱼₘ
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


# =============================================================================
# Unit test
# =============================================================================

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
