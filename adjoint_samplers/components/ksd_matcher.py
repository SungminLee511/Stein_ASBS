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
        darw_beta: float = 0.0,
        darw_weight_clip: float = 10.0,
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
        self.darw_beta = darw_beta
        self.darw_weight_clip = darw_weight_clip

        # Logging
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0
        self._last_darw_weight_max = 0.0
        self._last_darw_weight_min = 0.0
        self._last_darw_weight_std = 0.0

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

        # Step 4: DARW weights
        if self.darw_beta > 0 and not is_asbs_init_stage and not in_warmup:
            darw_weights = self._compute_darw_weights(x1)
        else:
            darw_weights = torch.ones(x1.shape[0], device=x1.device)

        # Step 5: Store in buffer
        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
            "darw_weights": darw_weights.to("cpu"),
        })

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1

        self._check_target_shape(t, xt, adjoint)

        if "darw_weights" in data:
            weights = data["darw_weights"].to(device)
        else:
            weights = torch.ones(x0.shape[0], device=device)

        return (t, xt), -adjoint, weights

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

    @torch.no_grad()
    def _compute_kde_chunked(
        self,
        x1: torch.Tensor,
        ell: torch.Tensor,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Compute KDE row means q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j) in chunks."""
        N = x1.shape[0]
        q_hat = torch.zeros(N, device=x1.device, dtype=x1.dtype)
        for i in range(0, N, chunk_size):
            x_chunk = x1[i:i + chunk_size]                    # (C, D)
            sq_dists = torch.cdist(x_chunk, x1).pow(2)        # (C, N)
            K_chunk = torch.exp(-sq_dists / (2 * ell ** 2))    # (C, N)
            q_hat[i:i + chunk_size] = K_chunk.mean(dim=1)
        return q_hat

    @torch.no_grad()
    def _compute_darw_weights(self, x1: torch.Tensor) -> torch.Tensor:
        """Compute DARW importance weights for the terminal batch.

        ŵ_i = (exp(-E(x1_i)) / q̂(x1_i))^β, self-normalized

        where q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j) is the KDE using the
        base kernel (same bandwidth as KSD).
        """
        N, D = x1.shape
        device = x1.device

        # 1. Bandwidth (same as KSD)
        if self.ksd_bandwidth is not None:
            ell = torch.tensor(self.ksd_bandwidth, device=device, dtype=x1.dtype)
        else:
            ell = median_bandwidth(x1)

        # 2. KDE: q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j)
        if N > self.ksd_efficient_threshold:
            q_hat = self._compute_kde_chunked(x1, ell)
        else:
            diffs = x1.unsqueeze(0) - x1.unsqueeze(1)       # (N, N, D)
            sq_dists = (diffs ** 2).sum(dim=-1)              # (N, N)
            K = torch.exp(-sq_dists / (2 * ell ** 2))        # (N, N)
            q_hat = K.mean(dim=1)                            # (N,)
        q_hat = q_hat.clamp(min=1e-10)

        # 3. Unnormalized target: p̃(x1_i) = exp(-E(x1_i))
        with torch.enable_grad():
            x1_req = x1.clone().detach().requires_grad_(True)
            energy_out = self.grad_term_cost.energy(x1_req)
        energies = energy_out["energy"].detach()             # (N,)

        # Stabilize: shift energies by min to avoid exp underflow
        energies = energies - energies.min()
        p_tilde = torch.exp(-energies)                       # (N,)

        # 4. Raw importance ratios
        ratios = p_tilde / q_hat                             # (N,)

        # 5. Soft power clipping with β
        ratios_beta = ratios ** self.darw_beta               # (N,)

        # 6. Hard clip for stability
        ratios_beta = ratios_beta.clamp(max=self.darw_weight_clip)

        # 7. Self-normalize so mean = 1
        weights = ratios_beta / ratios_beta.mean()           # (N,), mean ≈ 1

        # Logging
        self._last_darw_weight_max = weights.max().item()
        self._last_darw_weight_min = weights.min().item()
        self._last_darw_weight_std = weights.std().item()

        return weights


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
        darw_beta: float = 0.0,
        darw_weight_clip: float = 10.0,
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
        self.darw_beta = darw_beta
        self.darw_weight_clip = darw_weight_clip
        self._last_ksd = 0.0
        self._last_ksd_grad_norm = 0.0
        self._last_darw_weight_max = 0.0
        self._last_darw_weight_min = 0.0
        self._last_darw_weight_std = 0.0

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

        # DARW weights
        if self.darw_beta > 0 and not is_asbs_init_stage and not in_warmup:
            darw_weights = KSDAdjointVEMatcher._compute_darw_weights(self, x1)
        else:
            darw_weights = torch.ones(x1.shape[0], device=x1.device)

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
            "darw_weights": darw_weights.to("cpu"),
        })

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1
        adjoint = adjoint * torch.exp(self.sde.ref_sde.coeff2(t))

        self._check_target_shape(t, xt, adjoint)

        if "darw_weights" in data:
            weights = data["darw_weights"].to(device)
        else:
            weights = torch.ones(x0.shape[0], device=device)

        return (t, xt), -adjoint, weights
