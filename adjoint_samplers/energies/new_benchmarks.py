"""
adjoint_samplers/energies/new_benchmarks.py

Three energy functions for mode concentration experiments:
- UnequalGMMEnergy: 2D, 5 modes with unequal weights (minority mode death test)
- ManyWell5DEnergy: 5D, 32 modes (combinatorial mode coverage)
- ManyWell32DEnergy: 32D, 65,536 modes (standard PIS/DDS/DGFS benchmark)
"""

import torch
import numpy as np
import itertools
from adjoint_samplers.energies.base_energy import BaseEnergy


class UnequalGMMEnergy(BaseEnergy):
    """5-mode 2D Gaussian mixture with unequal weights.

    Weights: [0.50, 0.25, 0.15, 0.07, 0.03]
    Mode centers placed at radius 5, equally spaced angles.
    sigma = 0.5 per mode.

    The 3% mode should vanish under AM mode concentration.
    SDR should preserve it.
    """
    def __init__(self, dim=2, device="cpu"):
        super().__init__("unequal_gmm", dim)
        assert dim == 2

        self.n_modes = 5
        self.weights = torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03])
        self.std = 0.5
        radius = 5.0

        angles = torch.linspace(0, 2 * np.pi, self.n_modes + 1)[:-1]
        self.centers = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1)  # (5, 2)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
            self.weights = self.weights.to(x.device)

        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, 5, 2)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, 5)
        log_components = -sq_dist / (2 * self.std ** 2) + torch.log(self.weights)
        return -torch.logsumexp(log_components, dim=1)

    def get_centers(self):
        return self.centers

    def get_weights(self):
        return self.weights

    def get_std(self):
        return self.std

    def get_ref_samples(self, n=10000):
        """Sample from the mixture by ancestral sampling."""
        # Sample mode indices according to weights
        mode_idx = torch.multinomial(self.weights.cpu(), n, replacement=True)  # (n,)
        # Sample from chosen Gaussian
        samples = self.centers.cpu()[mode_idx] + self.std * torch.randn(n, 2)
        return samples


class ManyWell5DEnergy(BaseEnergy):
    """5D Many-Well: product of 5 independent 1D double-well potentials.

    E(x) = sum_{i=1}^{5} E_DW(x_i)
    E_DW(a) = a^4 - 6a^2 - 0.5a

    Each 1D double-well has 2 modes (left well ~-1.7, right well ~1.7).
    Total: 2^5 = 32 modes.
    All modes have approximately equal weight (broken by 0.5a term).

    This is the MW5 benchmark from the ASBS paper.
    """
    def __init__(self, dim=5, device="cpu"):
        super().__init__("mw5", dim)
        assert dim == 5
        self.n_modes = 32  # 2^5

    def _double_well_1d(self, a):
        """E_DW(a) = a^4 - 6a^2 - 0.5a"""
        return a ** 4 - 6.0 * a ** 2 - 0.5 * a

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 5) -> E: (B,)"""
        E = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(self.dim):
            E = E + self._double_well_1d(x[:, i])
        return E

    def get_mode_centers(self):
        """Return all 2^5 = 32 mode center coordinates.

        Each dimension has two wells. Find them by minimizing E_DW(a).
        E_DW'(a) = 4a^3 - 12a - 0.5 = 0
        Approximate: left well ~ -1.745, right well ~ 1.762
        """
        # Find wells numerically
        a = torch.linspace(-3, 3, 1000)
        e = a ** 4 - 6 * a ** 2 - 0.5 * a
        # Left well
        left_idx = e[:500].argmin()
        left_well = a[left_idx].item()
        # Right well
        right_idx = e[500:].argmin() + 500
        right_well = a[right_idx].item()

        wells = [left_well, right_well]
        # All 2^5 combinations
        centers = torch.tensor(list(itertools.product(wells, repeat=5)))
        return centers  # (32, 5)

    def get_ref_samples(self, n=10000):
        """Each dimension is independent 1D double-well.
        Use MCMC per dimension, then combine."""
        all_dims = []
        for d in range(5):
            s = self._sample_1d_double_well(n, n_burnin=5000)
            all_dims.append(s)
        samples = torch.stack(all_dims, dim=1)  # (n, 5)
        return samples

    @staticmethod
    def _sample_1d_double_well(n_samples, n_burnin=5000):
        """Sample from p(a) ~ exp(-(a^4 - 6a^2 - 0.5a)) using inverse CDF
        on a fine grid. Exact up to grid resolution, no MCMC mixing issues.

        The energy barrier between wells is too high for a single MCMC chain
        to cross, so we use a grid-based approach instead.
        """
        # Fine grid covering the support (tails decay as a^4)
        n_grid = 200000
        a_grid = torch.linspace(-5, 5, n_grid)
        da = a_grid[1] - a_grid[0]

        # Unnormalized log density
        log_p = -(a_grid ** 4 - 6 * a_grid ** 2 - 0.5 * a_grid)
        log_p = log_p - log_p.max()  # numerical stability
        p = torch.exp(log_p)

        # Build CDF
        cdf = torch.cumsum(p * da, dim=0)
        cdf = cdf / cdf[-1]  # normalize to [0, 1]

        # Inverse CDF sampling
        u = torch.rand(n_samples)
        # searchsorted: find indices where u would be inserted in cdf
        indices = torch.searchsorted(cdf, u)
        indices = indices.clamp(0, n_grid - 1)
        samples = a_grid[indices]

        # Add sub-grid jitter (uniform within grid cell) to avoid discretization
        samples = samples + (torch.rand(n_samples) - 0.5) * da

        return samples


class ManyWell32DEnergy(BaseEnergy):
    """32D Many-Well: product of 16 independent 2D double-well potentials.

    From DGFS (Zhang et al., ICLR 2024), PIS (Zhang & Chen, 2022).

    E(x) = sum_{i=1}^{16} E_DW(x_{2i-1}, x_{2i})
    E_DW(a, b) = a^4 - 6a^2 - 0.5a + 0.5b^2

    Each 2D pair: a-dimension has 2 modes (double-well), b-dimension is Gaussian.
    Total: 2^16 = 65,536 modes.
    """
    def __init__(self, dim=32, device="cpu"):
        super().__init__("manywell32", dim)
        assert dim == 32
        self.n_pairs = 16
        self.n_modes = 2 ** 16  # 65,536

    def _double_well_2d(self, a, b):
        """E_DW(a, b) = a^4 - 6a^2 - 0.5a + 0.5b^2"""
        return a ** 4 - 6.0 * a ** 2 - 0.5 * a + 0.5 * b ** 2

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 32) -> E: (B,)"""
        E = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(self.n_pairs):
            a = x[:, 2 * i]
            b = x[:, 2 * i + 1]
            E = E + self._double_well_2d(a, b)
        return E

    def get_1d_wells(self):
        """Return the two well positions in the a-dimension."""
        a = torch.linspace(-3, 3, 1000)
        e = a ** 4 - 6 * a ** 2 - 0.5 * a
        left_idx = e[:500].argmin()
        right_idx = e[500:].argmin() + 500
        return a[left_idx].item(), a[right_idx].item()

    def get_ref_samples(self, n=10000):
        """Each 2D pair is independent. a-dim: MCMC. b-dim: Gaussian."""
        all_pairs = []
        for pair in range(16):
            a_samples = ManyWell5DEnergy._sample_1d_double_well(n, n_burnin=5000)
            b_samples = torch.randn(n)  # b ~ N(0, 1) since E_b = 0.5*b^2
            all_pairs.extend([a_samples, b_samples])
        samples = torch.stack(all_pairs, dim=1)  # (n, 32)
        return samples
