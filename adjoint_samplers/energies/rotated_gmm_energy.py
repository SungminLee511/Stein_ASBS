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

        # Place modes on a ring in first 2 dims, with z-perturbation
        centers_low = np.zeros((n_modes, dim))
        for k in range(n_modes):
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

        self._centers = torch.tensor(centers_rotated, dtype=torch.float32)
        self._rotation = torch.tensor(Q, dtype=torch.float32)

        # Equal weights (log space)
        self._log_weights = torch.zeros(n_modes)

        # Precompute reference samples for evaluation
        ref_samples = []
        n_per_mode = 2000 // n_modes
        for k in range(n_modes):
            samples_k = (
                torch.randn(n_per_mode, dim) * mode_std
                + torch.tensor(centers_rotated[k], dtype=torch.float32)
            )
            ref_samples.append(samples_k)
        self._ref_samples = torch.cat(ref_samples, dim=0)

    def _to_device(self, device):
        self._centers = self._centers.to(device)
        self._rotation = self._rotation.to(device)
        self._log_weights = self._log_weights.to(device)
        self._ref_samples = self._ref_samples.to(device)
        self.device = device

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x) = -log Σ_k w_k N(x; μ_k, σ²I).

        Args:
            x: (B, D) sample positions

        Returns:
            E: (B,) energies
        """
        if self._centers.device != x.device:
            self._to_device(x.device)

        # (B, K, D): x_i - μ_k
        diff = x.unsqueeze(1) - self._centers.unsqueeze(0)  # (B, K, D)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, K)

        # log N(x; μ_k, σ²I) = -||x-μ_k||²/(2σ²)  (drop constant)
        log_probs = -sq_dist / (2 * self.mode_std ** 2)  # (B, K)
        log_probs = log_probs + self._log_weights.unsqueeze(0)

        # log-sum-exp for numerical stability
        log_mixture = torch.logsumexp(log_probs, dim=1)  # (B,)

        # E = -log p (up to constant)
        return -log_mixture

    def get_ref_samples(self):
        """Return precomputed reference samples for evaluation."""
        return self._ref_samples

    def get_mode_centers(self):
        """Return the rotated mode centers."""
        return self._centers

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
        if self._centers.device != samples.device:
            self._to_device(samples.device)

        threshold = threshold_factor * self.mode_std
        centers = self._centers  # (K, D)

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
