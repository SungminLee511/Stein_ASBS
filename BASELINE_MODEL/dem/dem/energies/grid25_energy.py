"""Grid25 Energy for DEM: 25-mode Gaussian mixture on a 5x5 grid.

Modes at {-4, -2, 0, 2, 4} x {-4, -2, 0, 2, 4}, each with sigma=0.3.
Equal weights (1/25).

p(x) ~ sum_{k=1}^{25} N(x; mu_k, 0.3^2 I)
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image


class Grid25Energy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        device="cpu",
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=10,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        self.std = 0.3
        self.n_modes = 25
        self.device = device
        self.curr_epoch = 0
        self.plot_samples_epoch_period = plot_samples_epoch_period
        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        self.data_path_train = data_path_train
        self.name = "grid25"

        # Build 5x5 grid centers
        coords = [-4.0, -2.0, 0.0, 2.0, 4.0]
        centers = []
        for i in coords:
            for j in coords:
                centers.append([i, j])
        self.centers = torch.tensor(centers, dtype=torch.float32, device=device)  # (25, 2)

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def _sample_from_modes(self, n: int) -> torch.Tensor:
        """Sample from the true Grid25 distribution (on self.device)."""
        K = self.centers.shape[0]
        n_per = n // K
        remainder = n - n_per * K
        samples = []
        for k in range(K):
            count = n_per + (1 if k < remainder else 0)
            s = torch.randn(count, 2, device=self.centers.device) * self.std + self.centers[k]
            samples.append(s)
        return torch.cat(samples, dim=0)

    def setup_test_set(self) -> Optional[torch.Tensor]:
        return self._sample_from_modes(self.test_set_size)

    def setup_train_set(self) -> Optional[torch.Tensor]:
        if self.data_path_train is not None:
            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu()
            else:
                data = torch.tensor(np.load(self.data_path_train, allow_pickle=True))
            return data
        return self.normalize(self._sample_from_modes(self.train_set_size))

    def setup_val_set(self) -> Optional[torch.Tensor]:
        return self._sample_from_modes(self.val_set_size)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """Return log probability (used as energy by DEM).

        samples: (N, 2) or (2,) for single sample (vmap compatibility).
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        if self.centers.device != samples.device:
            self.centers = self.centers.to(samples.device)

        # Handle both batched and single-sample (vmap) cases
        if samples.dim() == 1:
            # Single sample from vmap: (2,) -> (1, 2) -> compute -> scalar
            samples = samples.unsqueeze(0)
            diff = samples.unsqueeze(1) - self.centers.unsqueeze(0)  # (1, 25, 2)
            sq_dist = (diff ** 2).sum(dim=-1)  # (1, 25)
            log_component = -sq_dist / (2 * self.std ** 2)
            # Proper log prob: logsumexp - log(K) - log(2*pi*sigma^2)
            log_norm = -np.log(self.n_modes) - np.log(2 * np.pi * self.std ** 2)
            return (torch.logsumexp(log_component, dim=1) + log_norm).squeeze(0)
        else:
            diff = samples.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, 25, 2)
            sq_dist = (diff ** 2).sum(dim=-1)  # (B, 25)
            log_component = -sq_dist / (2 * self.std ** 2)
            log_norm = -np.log(self.n_modes) - np.log(2 * np.pi * self.std ** 2)
            return torch.logsumexp(log_component, dim=1) + log_norm

    @property
    def dimensionality(self):
        return 2

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize and latest_samples is not None:
                latest_samples = self.unnormalize(latest_samples)

            if latest_samples is not None:
                fig, ax = plt.subplots(figsize=(8, 8))

                # Plot contours of log prob
                bounds = (-6, 6)
                grid_n = 200
                xx = torch.linspace(bounds[0], bounds[1], grid_n)
                yy = torch.linspace(bounds[0], bounds[1], grid_n)
                XX, YY = torch.meshgrid(xx, yy, indexing="ij")
                grid_pts = torch.stack([XX.flatten(), YY.flatten()], dim=-1)
                with torch.no_grad():
                    # Temporarily disable unnormalize for contour plotting
                    old_unnorm = self.should_unnormalize
                    self.should_unnormalize = False
                    log_probs = self(grid_pts.to(self.centers.device)).cpu()
                    self.should_unnormalize = old_unnorm
                ZZ = log_probs.reshape(grid_n, grid_n)
                ax.contourf(
                    XX.numpy(), YY.numpy(), ZZ.numpy(),
                    levels=50, cmap="viridis", alpha=0.6,
                )

                # Plot samples
                s = latest_samples.detach().cpu()
                ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.5, c="red")
                ax.set_xlim(*bounds)
                ax.set_ylim(*bounds)
                ax.set_title(f"Grid25 - Epoch {self.curr_epoch}")
                ax.set_aspect("equal")

                wandb_logger.log_image(
                    f"{prefix}generated_samples", [fig_to_image(fig)]
                )
                plt.close(fig)

        self.curr_epoch += 1
