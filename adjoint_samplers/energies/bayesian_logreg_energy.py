"""
adjoint_samplers/energies/bayesian_logreg_energy.py

Bayesian logistic regression posterior as an energy function.
E(theta) = -Sigma_i log sigma(y_i theta^T x_i) + (lambda/2)||theta||^2

The posterior p(theta|data) is proportional to exp(-E(theta)) — a high-dimensional
distribution with no known collective variables.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class BayesianLogRegEnergy(BaseEnergy):
    """Bayesian logistic regression posterior.

    Args:
        dim: number of features
        dataset: 'australian' (d=15) or 'german' (d=25)
        prior_scale: prior std for weights (lambda = 1/prior_scale^2)
        device: torch device
    """
    def __init__(
        self,
        dim: int,
        dataset: str = 'australian',
        prior_scale: float = 1.0,
        device: str = 'cpu',
    ):
        super().__init__(f"blogreg_{dataset}", dim)
        self.prior_precision = 1.0 / (prior_scale ** 2)
        self.device = device

        # Load or generate data
        X, y = self._load_data(dataset, dim)
        self._X = torch.tensor(X, dtype=torch.float32)  # (N_data, dim)
        self._y = torch.tensor(y, dtype=torch.float32)   # (N_data,)
        self.N_data = self._X.shape[0]

        # Reference samples generated lazily (HMC is slow — only run at eval time)
        self._ref_samples = None

    def _load_data(self, dataset, dim):
        """Load or generate classification data."""
        try:
            from sklearn.datasets import load_breast_cancer
            from sklearn.preprocessing import StandardScaler
            if dataset == 'australian':
                # Use first `dim` features of breast cancer dataset
                data = load_breast_cancer()
                X = StandardScaler().fit_transform(data.data[:, :dim])
                y = 2.0 * data.target - 1.0  # Convert to {-1, +1}
            elif dataset == 'german':
                # Synthetic 25D classification
                rng = np.random.RandomState(42)
                N = 500
                X = rng.randn(N, dim)
                w_true = rng.randn(dim) * 0.5
                logits = X @ w_true
                y = 2.0 * (logits > 0).astype(float) - 1.0
                flip = rng.rand(N) < 0.1
                y[flip] *= -1
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
        except ImportError:
            # Fallback: synthetic data
            rng = np.random.RandomState(42)
            N = 500
            X = rng.randn(N, dim)
            w_true = rng.randn(dim) * 0.5
            logits = X @ w_true
            y = 2.0 * (logits > 0).astype(float) - 1.0
            flip = rng.rand(N) < 0.1
            y[flip] *= -1

        return X.astype(np.float32), y.astype(np.float32)

    @torch.no_grad()
    def _generate_reference(self, n_samples=2000, n_burnin=1000):
        """Generate reference samples via HMC chain.

        Uses precomputed X^T y and X^T X for fast gradient computation
        to avoid expensive autograd calls.
        Must run under torch.no_grad() to prevent graph accumulation.
        """
        dim = self.dim
        theta = torch.zeros(dim)
        samples = []
        step_size = 0.05
        n_leapfrog = 10

        X = self._X  # (N_data, dim)
        y = self._y   # (N_data,)
        lam = self.prior_precision
        n_accept = 0

        def energy_and_grad(th):
            """Fast energy + gradient without autograd."""
            logits = X @ th  # (N_data,)
            y_logits = y * logits
            # E = -sum log sigma(y*logit) + (lam/2)||th||^2
            E = -torch.nn.functional.logsigmoid(y_logits).sum() + 0.5 * lam * (th ** 2).sum()
            # grad E = -X^T (y * sigma(-y*logit)) + lam * th
            sig = torch.sigmoid(-y_logits)  # 1 - sigma(y*logit)
            grad = -(X.T @ (y * sig)) + lam * th
            return E, grad

        for i in range(n_burnin + n_samples):
            p = torch.randn(dim)
            theta_new = theta.clone()
            p_new = p.clone()

            # Leapfrog
            _, g = energy_and_grad(theta_new)
            p_new = p_new - 0.5 * step_size * g
            for _ in range(n_leapfrog - 1):
                theta_new = theta_new + step_size * p_new
                _, g = energy_and_grad(theta_new)
                p_new = p_new - step_size * g
            theta_new = theta_new + step_size * p_new
            _, g = energy_and_grad(theta_new)
            p_new = p_new - 0.5 * step_size * g

            # Accept/reject
            E_old, _ = energy_and_grad(theta)
            E_new, _ = energy_and_grad(theta_new)
            dH = (E_new - E_old) + 0.5 * (p_new.norm()**2 - p.norm()**2)
            if torch.rand(1).item() < torch.exp(-dH).clamp(max=1).item():
                theta = theta_new
                n_accept += 1

            if i >= n_burnin:
                samples.append(theta.clone())

        return torch.stack(samples)

    def _eval_single(self, theta, X, y):
        """Compute energy for a single theta vector."""
        logits = X @ theta  # (N_data,)
        log_likelihood = torch.nn.functional.logsigmoid(y * logits).sum()
        log_prior = -0.5 * self.prior_precision * (theta ** 2).sum()
        return -(log_likelihood + log_prior)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for a batch. x: (B, dim)."""
        if self._X.device != x.device:
            self._X = self._X.to(x.device)
            self._y = self._y.to(x.device)

        # Vectorized: x is (B, dim), X is (N_data, dim)
        logits = x @ self._X.T  # (B, N_data)
        log_lik = torch.nn.functional.logsigmoid(
            self._y.unsqueeze(0) * logits  # (B, N_data)
        ).sum(dim=1)  # (B,)
        log_prior = -0.5 * self.prior_precision * (x ** 2).sum(dim=1)
        return -(log_lik + log_prior)  # (B,)

    def get_ref_samples(self):
        """Lazy HMC reference generation — only runs on first call."""
        if self._ref_samples is None:
            print("[BayesianLogRegEnergy] Generating HMC reference samples (one-time cost)...")
            self._ref_samples = self._generate_reference(n_samples=2000)
            print(f"[BayesianLogRegEnergy] Done. Generated {self._ref_samples.shape[0]} samples.")
        return self._ref_samples
