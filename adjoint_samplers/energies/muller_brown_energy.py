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

        # Precompute reference samples via rejection sampling
        self._ref_samples = self._precompute_reference()

    def _precompute_reference(self):
        """Generate reference samples by importance resampling on a grid."""
        x = torch.linspace(-1.5, 1.2, 200)
        y = torch.linspace(-0.5, 2.0, 200)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        E = self._eval_raw(grid)
        log_p = -E / self.temperature
        log_p = log_p - log_p.max()
        p = torch.exp(log_p)
        p = p / p.sum()

        # Resample
        idx = torch.multinomial(p, 5000, replacement=True)
        return grid[idx]

    def _eval_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw Müller-Brown energy (no temperature scaling). x: (B, 2)."""
        x1 = x[..., 0]
        x2 = x[..., 1]

        result = torch.zeros_like(x1)
        for k in range(4):
            result = result + self.A[k] * torch.exp(
                self.a[k] * (x1 - self.x0[k])**2
                + self.b[k] * (x1 - self.x0[k]) * (x2 - self.y0[k])
                + self.c[k] * (x2 - self.y0[k])**2
            )
        return result

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy scaled by temperature. x: (B, 2)."""
        assert x.shape[-1] == 2
        return self._eval_raw(x) / self.temperature

    def get_ref_samples(self):
        return self._ref_samples

    def plot_landscape(self, samples=None, samples_sdr=None, save_path=None):
        """Plot the 2D energy landscape with optional sample overlays."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = torch.linspace(-1.5, 1.2, 300)
        y = torch.linspace(-0.5, 2.0, 300)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        E = self._eval_raw(grid).reshape(300, 300)

        n_panels = 1 + (samples is not None) + (samples_sdr is not None)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        for ax in axes:
            ax.contourf(xx.numpy(), yy.numpy(), E.numpy(),
                       levels=50, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        idx = 0
        axes[idx].set_title('Reference Samples')
        ref = self._ref_samples[:500]
        axes[idx].scatter(ref[:, 0], ref[:, 1], s=1, c='white', alpha=0.5)
        idx += 1

        if samples is not None:
            axes[idx].set_title('Baseline ASBS')
            s = samples[:500].cpu()
            axes[idx].scatter(s[:, 0], s[:, 1], s=3, c='red', alpha=0.6)
            idx += 1

        if samples_sdr is not None:
            axes[idx].set_title('SDR-ASBS')
            s = samples_sdr[:500].cpu()
            axes[idx].scatter(s[:, 0], s[:, 1], s=3, c='orange', alpha=0.6)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
