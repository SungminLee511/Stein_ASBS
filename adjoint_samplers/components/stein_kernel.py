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
    kernel: str = "rbf",
) -> torch.Tensor:
    """Compute KSD²(q, p) using the V-statistic estimator.

    KSD²_V = (1/N²) Σᵢⱼ kₚ(xᵢ, xⱼ)

    where kₚ is the Stein kernel with the chosen base kernel.

    Args:
        samples: (N, D) terminal particle positions.
        scores: (N, D) score sₚ(xᵢ) = -∇E(xᵢ) at each particle.
        ell: scalar bandwidth. If None, uses median heuristic.
        kernel: "rbf" or "imq".

    Returns:
        Scalar tensor: KSD² estimate.
    """
    N, D = samples.shape

    if ell is None:
        ell = median_bandwidth(samples)

    # Pairwise differences: δ[i,j] = xᵢ - xⱼ
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dists = (delta ** 2).sum(dim=-1)  # (N, N)

    # Score dot products
    ss = scores @ scores.T  # (N, N)
    s_dot_delta = torch.einsum('id,ijd->ij', scores, delta)  # (N, N)
    sp_dot_delta = torch.einsum('jd,ijd->ij', scores, delta)  # (N, N)

    if kernel == "rbf":
        kp = _stein_kernel_rbf(sq_dists, ss, s_dot_delta, sp_dot_delta, ell, D)
    elif kernel == "imq":
        kp = _stein_kernel_imq(sq_dists, ss, s_dot_delta, sp_dot_delta, ell, D)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Choose 'rbf' or 'imq'.")

    # V-statistic: (1/N²) Σᵢⱼ kₚ(xᵢ, xⱼ)
    return kp.sum() / (N ** 2)


def _stein_kernel_rbf(sq_dists, ss, s_dot_delta, sp_dot_delta, ell, D):
    """Compute Stein kernel matrix with RBF base kernel.

    k(x,x') = exp(-‖x-x'‖² / 2ℓ²)
    ∇ₓ k = -k · (x-x')/ℓ²
    ∇ₓ' k = k · (x-x')/ℓ²
    ∇²ₓₓ' k = k · [D/ℓ² - ‖x-x'‖²/ℓ⁴]

    Stein kernel: kₚ = k · [s^Ts' + s^Tδ/ℓ² - s'^Tδ/ℓ² + D/ℓ² - r²/ℓ⁴]
    """
    ell2 = ell ** 2
    ell4 = ell ** 4
    K = torch.exp(-sq_dists / (2 * ell2))
    Gamma = ss + s_dot_delta / ell2 - sp_dot_delta / ell2 + D / ell2 - sq_dists / ell4
    return K * Gamma


def _stein_kernel_imq(sq_dists, ss, s_dot_delta, sp_dot_delta, ell, D):
    """Compute Stein kernel matrix with IMQ base kernel.

    k(x,x') = (c² + ‖x-x'‖²)^{-1/2},  where c = ℓ

    ∇ₓ k  = -(c² + r²)^{-3/2} · (x-x')
    ∇ₓ' k =  (c² + r²)^{-3/2} · (x-x')      [opposite sign]
    ∇²ₓₓ' k = (c² + r²)^{-3/2} · D - 3(c² + r²)^{-5/2} · r²

    Stein kernel:
        kₚ = k · s^Ts'
           + (c²+r²)^{-3/2} · [s^T(x'-x) - s'^T(x'-x)]     [= (c²+r²)^{-3/2} · (-s^Tδ + s'^Tδ)]
           + (c²+r²)^{-3/2} · D - 3(c²+r²)^{-5/2} · r²

    where δ = x - x'.

    Note: using ℓ as c (the IMQ scale parameter) for config compatibility.
    """
    c2 = ell ** 2
    c2_r2 = c2 + sq_dists                       # (N, N)
    k = c2_r2 ** (-0.5)                          # (N, N)
    c2_r2_inv32 = c2_r2 ** (-1.5)                # (N, N)
    c2_r2_inv52 = c2_r2 ** (-2.5)                # (N, N)

    # Term 1: k · s^T s'
    term1 = k * ss

    # Term 2: (c²+r²)^{-3/2} · [-s^Tδ + s'^Tδ]
    # δ = x - x', so s^T(x'-x) = -s^Tδ
    term2 = c2_r2_inv32 * (-s_dot_delta + sp_dot_delta)

    # Term 3: trace of ∇²ₓₓ' k = D·(c²+r²)^{-3/2} - 3r²·(c²+r²)^{-5/2}
    term3 = D * c2_r2_inv32 - 3 * sq_dists * c2_r2_inv52

    return term1 + term2 + term3


# =============================================================================
# Stein kernel gradient (detached, no Hessian)
# =============================================================================

@torch.no_grad()
def compute_stein_kernel_gradient(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Compute Σⱼ ∇ₓ kₚ(xᵢ, xⱼ) for each particle i (detached, no Hessian).

    Args:
        samples: (N, D) terminal particle positions.
        scores: (N, D) score sₚ(xᵢ) = -∇E(xᵢ).
        ell: scalar bandwidth (RBF ℓ or IMQ c).
        kernel: "rbf" or "imq".

    Returns:
        grad_sum: (N, D) — Σⱼ ∇ₓ kₚ(xᵢ, xⱼ) for each i.
    """
    N, D = samples.shape

    # Pairwise: δ[i,j,d] = xᵢ^d - xⱼ^d
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)  # (N, N, D)
    sq_dists = (delta ** 2).sum(dim=-1)  # (N, N)

    # Score dot products
    ss = scores @ scores.T  # (N, N)
    s_dot_delta = torch.einsum('id,ijd->ij', scores, delta)  # (N, N)
    sp_dot_delta = torch.einsum('jd,ijd->ij', scores, delta)  # (N, N)

    # Score differences
    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (N, N, D)

    if kernel == "rbf":
        grad_sum = _stein_grad_rbf(delta, sq_dists, ss, s_dot_delta, sp_dot_delta, s_diff, ell, D)
    elif kernel == "imq":
        grad_sum = _stein_grad_imq(delta, sq_dists, ss, s_dot_delta, sp_dot_delta, s_diff, ell, D)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Choose 'rbf' or 'imq'.")

    return grad_sum


def _stein_grad_rbf(delta, sq_dists, ss, s_dot_delta, sp_dot_delta, s_diff, ell, D):
    """Gradient of RBF Stein kernel, summed over j.

    ∇ₓᵢ kₚ(xᵢ, xⱼ) = K · [-δ/ℓ² · Γ + (sᵢ-sⱼ)/ℓ² - 2δ/ℓ⁴]
    """
    ell2 = ell ** 2
    ell4 = ell ** 4
    K = torch.exp(-sq_dists / (2 * ell2))  # (N, N)
    Gamma = ss + s_dot_delta / ell2 - sp_dot_delta / ell2 + D / ell2 - sq_dists / ell4
    Gamma = Gamma.clamp(-1e4, 1e4)  # prevent overflow in partA

    partA = -delta / ell2 * Gamma.unsqueeze(-1)
    partB = s_diff / ell2
    partC = -2 * delta / ell4

    grad_kp = K.unsqueeze(-1) * (partA + partB + partC)  # (N, N, D)
    grad_kp = torch.nan_to_num(grad_kp, nan=0.0, posinf=0.0, neginf=0.0)
    return grad_kp.sum(dim=1)


def _stein_grad_imq(delta, sq_dists, ss, s_dot_delta, sp_dot_delta, s_diff, ell, D):
    """Gradient of IMQ Stein kernel, summed over j.

    IMQ: k = (c² + r²)^{-1/2},  c = ℓ

    The Stein kernel kₚ = term1 + term2 + term3:
      term1 = k · s^Ts'
      term2 = (c²+r²)^{-3/2} · (-s^Tδ + s'^Tδ)
      term3 = D·(c²+r²)^{-3/2} - 3r²·(c²+r²)^{-5/2}

    Differentiating each w.r.t. xᵢ (δ = xᵢ - xⱼ, ∂δ/∂xᵢ = I, ∂r²/∂xᵢ = 2δ):

      ∇ term1 = (∇k) · s^Ts'  = -(c²+r²)^{-3/2} · δ · s^Ts'
      ∇ term2 = -3(c²+r²)^{-5/2} · 2δ · (-s^Tδ + s'^Tδ)
                + (c²+r²)^{-3/2} · (-sᵢ + sⱼ)      [detached, no ∇s term]
      ∇ term3 = -3(c²+r²)^{-5/2} · 2δ · D
                - 3 · [2δ · (c²+r²)^{-5/2} + r² · (-5/2)(c²+r²)^{-7/2} · 2δ]
              = -6D · δ · (c²+r²)^{-5/2}
                - 6δ · (c²+r²)^{-5/2} + 15r² · δ · (c²+r²)^{-7/2}
              = -6(D+1) · δ · (c²+r²)^{-5/2} + 15r² · δ · (c²+r²)^{-7/2}
    """
    c2 = ell ** 2
    c2_r2 = c2 + sq_dists                       # (N, N)
    c2_r2_inv32 = c2_r2 ** (-1.5)
    c2_r2_inv52 = c2_r2 ** (-2.5)
    c2_r2_inv72 = c2_r2 ** (-3.5)

    # ∇ term1: -(c²+r²)^{-3/2} · δ · s^Ts'
    grad1 = -(c2_r2_inv32 * ss).unsqueeze(-1) * delta  # (N, N, D)

    # ∇ term2:
    #   -6(c²+r²)^{-5/2} · δ · (-s^Tδ + s'^Tδ)
    #   + (c²+r²)^{-3/2} · (-sᵢ + sⱼ)
    bracket2 = -s_dot_delta + sp_dot_delta  # (N, N)
    grad2_a = (-6 * c2_r2_inv52 * bracket2).unsqueeze(-1) * delta
    grad2_b = c2_r2_inv32.unsqueeze(-1) * (-s_diff)  # -sᵢ+sⱼ = -(sᵢ-sⱼ)
    grad2 = grad2_a + grad2_b

    # ∇ term3: -6(D+1) · δ · (c²+r²)^{-5/2} + 15r² · δ · (c²+r²)^{-7/2}
    grad3 = (-6 * (D + 1) * c2_r2_inv52 + 15 * sq_dists * c2_r2_inv72).unsqueeze(-1) * delta

    grad_kp = grad1 + grad2 + grad3  # (N, N, D)
    return grad_kp.sum(dim=1)


# =============================================================================
# Memory-efficient version for large N (avoids N×N×D tensors)
# =============================================================================

@torch.no_grad()
def compute_stein_kernel_gradient_efficient(
    samples: torch.Tensor,
    scores: torch.Tensor,
    ell: torch.Tensor,
    chunk_size: int = 256,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Memory-efficient version of compute_stein_kernel_gradient.

    Processes pairs in chunks to avoid materializing (N, N, D) tensors.

    Args:
        samples: (N, D)
        scores: (N, D)
        ell: scalar bandwidth
        chunk_size: number of j particles per chunk
        kernel: "rbf" or "imq"

    Returns:
        grad_sum: (N, D)
    """
    N, D = samples.shape
    grad_sum = torch.zeros_like(samples)

    for j_start in range(0, N, chunk_size):
        j_end = min(j_start + chunk_size, N)

        x_j = samples[j_start:j_end]  # (M, D)
        s_j = scores[j_start:j_end]   # (M, D)

        # δ[i, m, d] = xᵢ - xⱼₘ
        delta = samples.unsqueeze(1) - x_j.unsqueeze(0)  # (N, M, D)
        sq_dists = (delta ** 2).sum(dim=-1)  # (N, M)

        ss = scores @ s_j.T  # (N, M)
        s_dot_d = torch.einsum('id,imd->im', scores, delta)
        sp_dot_d = torch.einsum('md,imd->im', s_j, delta)
        s_diff = scores.unsqueeze(1) - s_j.unsqueeze(0)  # (N, M, D)

        if kernel == "rbf":
            ell2 = ell ** 2
            ell4 = ell ** 4
            K = torch.exp(-sq_dists / (2 * ell2))
            Gamma = ss + s_dot_d / ell2 - sp_dot_d / ell2 + D / ell2 - sq_dists / ell4
            Gamma = Gamma.clamp(-1e4, 1e4)

            partA = -delta / ell2 * Gamma.unsqueeze(-1)
            partB = s_diff / ell2
            partC = -2 * delta / ell4

            grad_chunk = K.unsqueeze(-1) * (partA + partB + partC)
            grad_chunk = torch.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0)

        elif kernel == "imq":
            c2 = ell ** 2
            c2_r2 = c2 + sq_dists
            c2_r2_inv32 = c2_r2 ** (-1.5)
            c2_r2_inv52 = c2_r2 ** (-2.5)
            c2_r2_inv72 = c2_r2 ** (-3.5)

            bracket2 = -s_dot_d + sp_dot_d

            grad1 = -(c2_r2_inv32 * ss).unsqueeze(-1) * delta
            grad2 = (-6 * c2_r2_inv52 * bracket2).unsqueeze(-1) * delta + \
                     c2_r2_inv32.unsqueeze(-1) * (-s_diff)
            grad3 = (-6 * (D + 1) * c2_r2_inv52 + 15 * sq_dists * c2_r2_inv72).unsqueeze(-1) * delta

            grad_chunk = grad1 + grad2 + grad3
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        grad_sum += grad_chunk.sum(dim=1)

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

    # === RBF tests ===
    print("=== RBF Kernel ===")
    ksd_true = compute_ksd_squared(x_true, s_true, kernel="rbf")
    print(f"KSD^2 (true samples from N(0,I)): {ksd_true.item():.6f}")

    x_biased = torch.randn(N, d) + 1.0
    s_biased = -x_biased
    ksd_biased = compute_ksd_squared(x_biased, s_biased, kernel="rbf")
    print(f"KSD^2 (biased samples, shift=1): {ksd_biased.item():.6f}")
    assert ksd_biased > ksd_true * 5, "KSD should be much larger for biased samples"

    ell = median_bandwidth(x_true)
    grad = compute_stein_kernel_gradient(x_true, s_true, ell, kernel="rbf")
    assert grad.shape == (N, d), f"Expected ({N}, {d}), got {grad.shape}"

    grad_eff = compute_stein_kernel_gradient_efficient(x_true, s_true, ell, chunk_size=128, kernel="rbf")
    diff = (grad - grad_eff).abs().max().item()
    print(f"Max diff (standard vs efficient, RBF): {diff:.2e}")
    assert diff < 1e-4, f"Efficient version diverges: {diff}"

    # === IMQ tests ===
    print("\n=== IMQ Kernel ===")
    ksd_true_imq = compute_ksd_squared(x_true, s_true, kernel="imq")
    print(f"KSD^2 (true samples from N(0,I)): {ksd_true_imq.item():.6f}")

    ksd_biased_imq = compute_ksd_squared(x_biased, s_biased, kernel="imq")
    print(f"KSD^2 (biased samples, shift=1): {ksd_biased_imq.item():.6f}")
    assert ksd_biased_imq > ksd_true_imq * 5, "IMQ KSD should be much larger for biased samples"

    grad_imq = compute_stein_kernel_gradient(x_true, s_true, ell, kernel="imq")
    assert grad_imq.shape == (N, d), f"Expected ({N}, {d}), got {grad_imq.shape}"

    grad_imq_eff = compute_stein_kernel_gradient_efficient(x_true, s_true, ell, chunk_size=128, kernel="imq")
    diff_imq = (grad_imq - grad_imq_eff).abs().max().item()
    print(f"Max diff (standard vs efficient, IMQ): {diff_imq:.2e}")
    assert diff_imq < 1e-4, f"IMQ efficient version diverges: {diff_imq}"

    # === Cross-kernel sanity: both should give finite, nonzero gradients ===
    print(f"\nRBF grad norm: {grad.norm():.4f}")
    print(f"IMQ grad norm: {grad_imq.norm():.4f}")
    assert grad.norm() > 0, "RBF grad should be nonzero"
    assert grad_imq.norm() > 0, "IMQ grad should be nonzero"

    print("\nAll Stein kernel tests passed! (RBF + IMQ)")


if __name__ == "__main__":
    _test_stein_kernel()
