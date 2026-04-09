# Copyright (c) SungminLee511. KSD-Augmented Adjoint Samplers.

"""
KSD-augmented adjoint matchers. Inherit from existing matchers and override
only the populate_buffer method to add the inter-particle Stein kernel
gradient correction to the adjoint terminal condition.

See .claude/skills/math_specs.md Section 4, 6 for the math.
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
        ksd_kernel: str = "rbf",
        ksd_score_beta: float = 1.0,
        ksd_imq_c: float | None = None,
        ksd_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ksd_lambda = ksd_lambda
        self.ksd_bandwidth = ksd_bandwidth
        self.ksd_max_particles = ksd_max_particles
        self.ksd_efficient_threshold = ksd_efficient_threshold
        self.ksd_kernel = ksd_kernel
        self.ksd_score_beta = ksd_score_beta
        self.ksd_imq_c = ksd_imq_c
        self.ksd_warmup_epochs = ksd_warmup_epochs

        # Logging
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
            epoch: int = -1,
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

        # Step 3: KSD correction — skip during init stage AND warmup
        in_warmup = (self.ksd_warmup_epochs > 0 and epoch >= 0
                     and epoch < self.ksd_warmup_epochs)
        if self.ksd_lambda > 0 and not is_asbs_init_stage and not in_warmup:
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

        From math_specs.md Section 4.1:
            Y₁ⁱ = -(1/N)∇Φ₀ - (λ/N²)Σⱼ ∇ₓkₚ

        The matcher stores adjoint1 = ∇Φ₀ (forces, positive sign) and
        negates it in prepare_target. So we add the KSD correction with
        the same sign convention:
            adjoint1_corrected = adjoint1 + (λ/N²) * Σⱼ ∇ₓkₚ
        """
        N, D = x1.shape
        device = x1.device

        # Subsample if too many particles
        if N > self.ksd_max_particles:
            idx = torch.randperm(N, device=device)[:self.ksd_max_particles]
            x1_sub = x1[idx]
        else:
            x1_sub = x1
            idx = None

        N_sub = x1_sub.shape[0]

        # Determine clip norm for scores and KSD correction
        if hasattr(self.grad_term_cost, 'max_grad_E_norm') and self.grad_term_cost.max_grad_E_norm is not None:
            clip_norm = self.grad_term_cost.max_grad_E_norm
        elif hasattr(self, '_ksd_clip_norm'):
            clip_norm = self._ksd_clip_norm
        else:
            clip_norm = None

        # Compute scores at terminal samples
        with torch.enable_grad():
            x1_req = x1_sub.clone().detach().requires_grad_(True)
            energy_out = self.grad_term_cost.energy(x1_req)
            scores = self.ksd_score_beta * energy_out["forces"].detach()

        # Sanitize scores: replace NaN/Inf with zero, then clip norms
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_norm is not None:
            s_norms = torch.linalg.vector_norm(scores, dim=-1, keepdim=True)
            s_clip = torch.clamp(clip_norm / (s_norms + 1e-6), max=1.0)
            scores = scores * s_clip

        # Bandwidth / IMQ scale parameter
        if self.ksd_imq_c is not None and self.ksd_kernel == "imq":
            # For IMQ kernel, ksd_imq_c overrides the bandwidth as the
            # scale parameter c in k(x,x') = (c² + ‖x-x'‖²)^{-1/2}
            ell = torch.tensor(self.ksd_imq_c, device=device, dtype=x1.dtype)
        elif self.ksd_bandwidth is not None:
            ell = torch.tensor(self.ksd_bandwidth, device=device, dtype=x1.dtype)
        else:
            ell = median_bandwidth(x1_sub)

        # Compute Stein kernel gradient sum
        if N_sub > self.ksd_efficient_threshold:
            grad_sum = compute_stein_kernel_gradient_efficient(
                x1_sub, scores, ell, chunk_size=256, kernel=self.ksd_kernel
            )
        else:
            grad_sum = compute_stein_kernel_gradient(
                x1_sub, scores, ell, kernel=self.ksd_kernel
            )

        # KSD correction: (λ / N²) * Σⱼ ∇ₓkₚ(xᵢ, xⱼ)
        ksd_correction = (self.ksd_lambda / (N_sub ** 2)) * grad_sum

        # Clip KSD correction per-sample
        if clip_norm is not None:
            norms = torch.linalg.vector_norm(ksd_correction, dim=-1, keepdim=True)
            clip_coeff = torch.clamp(clip_norm / (norms + 1e-6), max=1.0)
            ksd_correction = ksd_correction * clip_coeff

        # Logging
        self._last_ksd = compute_ksd_squared(
            x1_sub, scores, ell, kernel=self.ksd_kernel
        ).item()
        self._last_ksd_grad_norm = ksd_correction.norm(dim=-1).mean().item()

        # If we subsampled, only correct the subset
        if idx is not None:
            full_correction = torch.zeros_like(adjoint1)
            full_correction[idx] = ksd_correction
            adjoint1 = adjoint1 + full_correction
        else:
            adjoint1 = adjoint1 + ksd_correction

        return adjoint1


class KSDAdjointVPMatcher(AdjointVPMatcher):
    """KSD-augmented matcher for VP-SDE. Same KSD logic, different base class."""
    def __init__(
        self,
        ksd_lambda: float = 1.0,
        ksd_bandwidth: float | None = None,
        ksd_max_particles: int = 2048,
        ksd_efficient_threshold: int = 1024,
        ksd_kernel: str = "rbf",
        ksd_score_beta: float = 1.0,
        ksd_imq_c: float | None = None,
        ksd_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ksd_lambda = ksd_lambda
        self.ksd_bandwidth = ksd_bandwidth
        self.ksd_max_particles = ksd_max_particles
        self.ksd_efficient_threshold = ksd_efficient_threshold
        self.ksd_kernel = ksd_kernel
        self.ksd_score_beta = ksd_score_beta
        self.ksd_imq_c = ksd_imq_c
        self.ksd_warmup_epochs = ksd_warmup_epochs
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
            epoch: int = -1,
    ):
        (x0, x1) = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=True,
        )
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        in_warmup = (self.ksd_warmup_epochs > 0 and epoch >= 0
                     and epoch < self.ksd_warmup_epochs)
        if self.ksd_lambda > 0 and not is_asbs_init_stage and not in_warmup:
            # Reuse KSDAdjointVEMatcher's correction logic
            adjoint1 = KSDAdjointVEMatcher._apply_ksd_correction(self, x1, adjoint1)

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
        })
