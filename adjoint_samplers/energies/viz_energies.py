"""
adjoint_samplers/energies/viz_energies.py

2D energy functions for visualization benchmarks.
All have known mode structure for quantitative evaluation.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class GMM9Energy(BaseEnergy):
    """9-mode Gaussian mixture on a 3x3 grid.

    From DGFS (Zhang et al., ICLR 2024).
    Modes at {-5, 0, 5} x {-5, 0, 5}, each with sigma=0.3.

    p(x) ~ sum_{k=1}^{9} N(x; mu_k, 0.3^2 I)
    """
    def __init__(self, dim=2, device="cpu"):
        super().__init__("gmm9", dim)
        assert dim == 2

        centers = []
        for i in [-5.0, 0.0, 5.0]:
            for j in [-5.0, 0.0, 5.0]:
                centers.append([i, j])
        self.centers = torch.tensor(centers, dtype=torch.float32)  # (9, 2)
        self.std = 0.3
        self.n_modes = 9

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) -> E: (B,)"""
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        # (B, 9, 2)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, 9)
        log_probs = -sq_dist / (2 * self.std ** 2)
        return -torch.logsumexp(log_probs, dim=1)

    def get_centers(self):
        return self.centers

    def get_std(self):
        return self.std

    def get_ref_samples(self, n=10000):
        """Generate reference samples by sampling uniformly from all modes."""
        K = self.centers.shape[0]
        n_per = n // K
        samples = []
        for k in range(K):
            s = torch.randn(n_per, 2) * self.std + self.centers[k]
            samples.append(s)
        return torch.cat(samples, dim=0)
