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


class TwoMoonsEnergy(BaseEnergy):
    """Two crescent-shaped modes that wrap around each other.

    Each moon is an arc of a circle with Gaussian noise perpendicular to the arc.
    Non-convex, interleaved geometry.
    """
    def __init__(self, dim=2, noise=0.1, device="cpu"):
        super().__init__("two_moons", dim)
        assert dim == 2
        self.noise = noise
        self.n_modes = 2
        # Approximate centers of the two crescents
        self._centers = torch.tensor([[0.0, 0.25], [1.0, -0.25]])

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Energy based on distance to nearest point on each crescent arc."""
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Moon 1: upper crescent centered at (0.5, 0), radius 1
        # Arc: angle from pi to 0
        angles1 = torch.atan2(x2, x1 - 0.5)
        r1 = torch.sqrt((x1 - 0.5) ** 2 + x2 ** 2)
        dist1 = (r1 - 1.0) ** 2

        # Moon 2: lower crescent centered at (0.5, 0), radius 1, flipped
        # Shift: (1-x1, -(x2 - 0.25))
        x1_flip = 1.0 - x1
        x2_flip = -(x2 - 0.25)
        angles2 = torch.atan2(x2_flip, x1_flip - 0.5)
        r2 = torch.sqrt((x1_flip - 0.5) ** 2 + x2_flip ** 2)
        dist2 = (r2 - 1.0) ** 2

        log_p1 = -dist1 / (2 * self.noise ** 2)
        log_p2 = -dist2 / (2 * self.noise ** 2)
        log_p = torch.logsumexp(torch.stack([log_p1, log_p2], dim=-1), dim=-1)
        return -log_p

    def get_centers(self):
        return self._centers

    def get_std(self):
        return 0.5

    def get_ref_samples(self, n=10000):
        """Generate reference samples from two moons distribution."""
        n_per = n // 2
        samples = []
        # Moon 1: upper crescent
        angles = torch.linspace(0, np.pi, n_per)
        x1 = 0.5 + torch.cos(angles) + torch.randn(n_per) * self.noise
        x2 = torch.sin(angles) + torch.randn(n_per) * self.noise
        samples.append(torch.stack([x1, x2], dim=1))
        # Moon 2: lower crescent (flipped and shifted)
        x1 = 0.5 - torch.cos(angles) + torch.randn(n_per) * self.noise
        x2 = -torch.sin(angles) + 0.25 + torch.randn(n_per) * self.noise
        samples.append(torch.stack([x1, x2], dim=1))
        return torch.cat(samples, dim=0)


class PinwheelEnergy(BaseEnergy):
    """5 elongated clusters arranged in a pinwheel/spiral pattern.

    Each arm is a thin stretched Gaussian rotated at a different angle.
    """
    def __init__(self, dim=2, n_arms=5, radial_std=0.3, tangential_std=0.05, device="cpu"):
        super().__init__("pinwheel", dim)
        assert dim == 2
        self.n_arms = n_arms
        self.n_modes = n_arms
        self.radial_std = radial_std
        self.tangential_std = tangential_std
        self.rate = 0.25  # spiral rate

        # Arm centers at radius ~2
        angles = torch.linspace(0, 2 * np.pi, n_arms + 1)[:-1]
        arm_radius = 2.0
        centers = torch.stack([arm_radius * torch.cos(angles),
                               arm_radius * torch.sin(angles)], dim=1)
        self._centers = centers
        self._angles = angles

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        if self._centers.device != x.device:
            self._centers = self._centers.to(x.device)
            self._angles = self._angles.to(x.device)

        log_probs = []
        for k in range(self.n_arms):
            angle = self._angles[k]
            # Rotation matrix
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            # Rotate x into arm-local coordinates
            dx = x[..., 0] - self._centers[k, 0]
            dy = x[..., 1] - self._centers[k, 1]
            local_r = cos_a * dx + sin_a * dy   # radial (elongated) direction
            local_t = -sin_a * dx + cos_a * dy  # tangential (thin) direction

            log_p = (-local_r ** 2 / (2 * self.radial_std ** 2)
                     - local_t ** 2 / (2 * self.tangential_std ** 2))
            log_probs.append(log_p)

        log_p = torch.logsumexp(torch.stack(log_probs, dim=-1), dim=-1)
        return -log_p

    def get_centers(self):
        return self._centers

    def get_std(self):
        return self.radial_std

    def get_ref_samples(self, n=10000):
        n_per = n // self.n_arms
        samples = []
        for k in range(self.n_arms):
            angle = self._angles[k].item()
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Sample in local coordinates
            local_r = torch.randn(n_per) * self.radial_std
            local_t = torch.randn(n_per) * self.tangential_std
            # Rotate back to global
            x1 = self._centers[k, 0] + cos_a * local_r - sin_a * local_t
            x2 = self._centers[k, 1] + sin_a * local_r + cos_a * local_t
            samples.append(torch.stack([x1, x2], dim=1))
        return torch.cat(samples, dim=0)


class CheckerboardEnergy(BaseEnergy):
    """4x4 checkerboard with alternating high/low density squares.

    8 "on" squares with smooth sigmoid boundaries. Grid spans [-4, 4]^2.
    """
    def __init__(self, dim=2, grid_size=4, sharpness=10.0, device="cpu"):
        super().__init__("checkerboard", dim)
        assert dim == 2
        self.grid_size = grid_size
        self.sharpness = sharpness  # sigmoid sharpness at boundaries
        self.cell_size = 8.0 / grid_size  # = 2.0 for 4x4 in [-4,4]
        self.n_modes = 8  # half of 16 squares

        # Centers of "on" squares
        centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    cx = -4.0 + (i + 0.5) * self.cell_size
                    cy = -4.0 + (j + 0.5) * self.cell_size
                    centers.append([cx, cy])
        self._centers = torch.tensor(centers, dtype=torch.float32)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth checkerboard via product of sigmoids."""
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Use sin to create checkerboard pattern, smoothed by sigmoid
        # sin(pi*x/cell) * sin(pi*y/cell) > 0 for "on" squares
        freq = np.pi / self.cell_size
        pattern = torch.sin(freq * x1) * torch.sin(freq * x2)
        log_p = torch.log(torch.sigmoid(self.sharpness * pattern) + 1e-10)

        # Add soft boundary to keep samples in [-4, 4]^2
        boundary = -0.1 * (torch.clamp(x1.abs() - 3.8, min=0) ** 2
                           + torch.clamp(x2.abs() - 3.8, min=0) ** 2)
        return -(log_p + boundary)

    def get_centers(self):
        return self._centers

    def get_std(self):
        return 0.5

    def get_ref_samples(self, n=10000):
        """Rejection sample from checkerboard."""
        samples = []
        n_collected = 0
        while n_collected < n:
            # Uniform in [-4, 4]^2
            x = torch.rand(n * 3, 2) * 8.0 - 4.0
            # Check if in "on" square
            i = ((x[:, 0] + 4.0) / self.cell_size).long().clamp(0, self.grid_size - 1)
            j = ((x[:, 1] + 4.0) / self.cell_size).long().clamp(0, self.grid_size - 1)
            on = ((i + j) % 2 == 0)
            samples.append(x[on])
            n_collected += on.sum().item()
        return torch.cat(samples, dim=0)[:n]


class NestedRingsEnergy(BaseEnergy):
    """Two concentric rings at r=2 and r=5 with weights 80/20.

    Inner ring is hard to find because the outer ring surrounds it.
    Each ring has Gaussian width.
    """
    def __init__(self, dim=2, r_inner=2.0, r_outer=5.0, width=0.3,
                 w_outer=0.8, device="cpu"):
        super().__init__("nested_rings", dim)
        assert dim == 2
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.width = width
        self.w_outer = w_outer
        self.w_inner = 1.0 - w_outer
        self.n_modes = 2

        # Approximate "centers" — points on each ring for visualization
        self._centers = torch.tensor([[r_inner, 0.0], [r_outer, 0.0]])

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt((x ** 2).sum(dim=-1))

        log_outer = -((r - self.r_outer) ** 2) / (2 * self.width ** 2) + np.log(self.w_outer)
        log_inner = -((r - self.r_inner) ** 2) / (2 * self.width ** 2) + np.log(self.w_inner)

        log_p = torch.logsumexp(torch.stack([log_outer, log_inner], dim=-1), dim=-1)
        return -log_p

    def get_centers(self):
        return self._centers

    def get_std(self):
        return self.width

    def get_ref_samples(self, n=10000):
        n_outer = int(n * self.w_outer)
        n_inner = n - n_outer
        samples = []
        # Outer ring
        angles = torch.rand(n_outer) * 2 * np.pi
        r = self.r_outer + torch.randn(n_outer) * self.width
        samples.append(torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1))
        # Inner ring
        angles = torch.rand(n_inner) * 2 * np.pi
        r = self.r_inner + torch.randn(n_inner) * self.width
        samples.append(torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1))
        return torch.cat(samples, dim=0)


class Grid25Energy(BaseEnergy):
    """25-mode Gaussian mixture on a 5x5 grid.

    Modes at {-4, -2, 0, 2, 4} x {-4, -2, 0, 2, 4}, each with sigma=0.3.
    Equal weights (1/25). Source N(0,I) sits directly on the center mode (0,0),
    creating a strong mode collapse trap for baseline ASBS.

    p(x) ~ sum_{k=1}^{25} N(x; mu_k, 0.3^2 I)
    """
    def __init__(self, dim=2, device="cpu"):
        super().__init__("grid25", dim)
        assert dim == 2

        centers = []
        for i in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            for j in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                centers.append([i, j])
        self.centers = torch.tensor(centers, dtype=torch.float32)  # (25, 2)
        self.std = 0.3
        self.n_modes = 25

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) -> E: (B,)"""
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        # (B, 25, 2)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, 25)
        log_probs = -sq_dist / (2 * self.std ** 2)
        return -torch.logsumexp(log_probs, dim=1)

    def get_centers(self):
        return self.centers

    def get_std(self):
        return self.std

    def get_ref_samples(self, n=10000):
        """Generate reference samples by sampling uniformly from all 25 modes."""
        K = self.centers.shape[0]
        n_per = n // K
        samples = []
        for k in range(K):
            s = torch.randn(n_per, 2) * self.std + self.centers[k]
            samples.append(s)
        return torch.cat(samples, dim=0)


class SpiralEnergy(BaseEnergy):
    """Single continuous Archimedean spiral density.

    Unimodal but complex non-convex geometry. Tests whether SDE can learn
    curved transport without mode collapse issues.
    """
    def __init__(self, dim=2, n_turns=2.0, width=0.15, device="cpu"):
        super().__init__("spiral", dim)
        assert dim == 2
        self.n_turns = n_turns
        self.width = width
        self.n_modes = 1  # unimodal
        self.max_angle = n_turns * 2 * np.pi

    def _nearest_spiral_dist_sq(self, x):
        """Compute squared distance from each point to nearest point on spiral.

        Spiral: r(t) = t/(2*pi*n_turns) * r_max, angle(t) = t
        for t in [0.5, max_angle]. We discretize and find nearest.
        """
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Discretize spiral
        n_pts = 500
        t = torch.linspace(0.5, self.max_angle, n_pts, device=x.device)
        r_max = 4.0
        r = t / self.max_angle * r_max
        sx = r * torch.cos(t)
        sy = r * torch.sin(t)

        # (B, n_pts)
        dx = x1.unsqueeze(-1) - sx.unsqueeze(0)
        dy = x2.unsqueeze(-1) - sy.unsqueeze(0)
        dist_sq = dx ** 2 + dy ** 2
        min_dist_sq = dist_sq.min(dim=-1).values
        return min_dist_sq

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        dist_sq = self._nearest_spiral_dist_sq(x)
        log_p = -dist_sq / (2 * self.width ** 2)
        return -log_p

    def get_centers(self):
        """Return points along the spiral for visualization."""
        t = torch.linspace(0.5, self.max_angle, 20)
        r_max = 4.0
        r = t / self.max_angle * r_max
        return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)

    def get_std(self):
        return self.width

    def get_ref_samples(self, n=10000):
        """Sample points along the spiral with Gaussian noise."""
        t = torch.rand(n) * (self.max_angle - 0.5) + 0.5
        r_max = 4.0
        r = t / self.max_angle * r_max
        sx = r * torch.cos(t)
        sy = r * torch.sin(t)
        # Add noise perpendicular to spiral direction
        noise = torch.randn(n) * self.width
        # Perpendicular direction: (-sin(t), cos(t))
        px = sx + noise * (-torch.sin(t))
        py = sy + noise * torch.cos(t)
        return torch.stack([px, py], dim=1)
