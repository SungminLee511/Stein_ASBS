# New Benchmarks: Implementation Guide for Claude Code

## Overview

Three new benchmarks to implement, train, and evaluate. All use FourierMLP controller, standard VESDE, Gaussian source. No graph structure (not particle systems).

|Benchmark         |dim|Modes |Key demonstration                               |
|------------------|---|------|------------------------------------------------|
|Unequal-Weight GMM|2  |5     |Minority mode death under AM, survival under SDR|
|Many-Well MW5     |5  |32    |Combinatorial mode coverage in low-d            |
|32D Many-Well     |32 |65,536|Standard benchmark from PIS/DDS/DGFS literature |

DO NOT modify any existing files. All new code goes in new files.

-----

## 1. Energy Functions

Create `adjoint_samplers/energies/new_benchmarks.py`:

```python
"""
adjoint_samplers/energies/new_benchmarks.py

Three energy functions for mode concentration experiments.
"""

import torch
import numpy as np
from adjoint_samplers.energies.base_energy import BaseEnergy


class UnequalGMMEnergy(BaseEnergy):
    """5-mode 2D Gaussian mixture with unequal weights.

    Weights: [0.50, 0.25, 0.15, 0.07, 0.03]
    Mode centers placed at radius 5, equally spaced angles.
    σ = 0.5 per mode.

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


class ManyWell5DEnergy(BaseEnergy):
    """5D Many-Well: product of 5 independent 1D double-well potentials.

    E(x) = sum_{i=1}^{5} E_DW(x_i)
    E_DW(a) = a^4 - 6a^2 - 0.5a

    Each 1D double-well has 2 modes (left well ~-1.7, right well ~1.7).
    Total: 2^5 = 32 modes.
    All modes have equal weight by symmetry (approximate, broken by 0.5a term).

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
        import itertools
        centers = torch.tensor(list(itertools.product(wells, repeat=5)))
        return centers  # (32, 5)


class ManyWell32DEnergy(BaseEnergy):
    """32D Many-Well: product of 16 independent 2D double-well potentials.

    From DGFS (Zhang et al., ICLR 2024), PIS (Zhang & Chen, 2022).

    E(x) = sum_{i=1}^{16} E_DW(x_{2i-1}, x_{2i})
    E_DW(a, b) = a^4 - 6a^2 - 0.5a + 0.5b^2    (NOTE: sign on b^2 makes b unimodal)

    Wait -- let me be precise. The standard definition from DGFS is:
    mu(a, b) = exp(-(a^4 - 6a^2 + 0.5a - 0.5b^2))
    So E_DW(a, b) = a^4 - 6a^2 + 0.5a - 0.5b^2
    But this makes E -> -inf as b -> inf (unstable).

    The correct DGFS definition uses unnormalized density:
    mu(a, b) = exp(-a^4 + 6a^2 + 0.5a - 0.5b^2)
    So E(a, b) = a^4 - 6a^2 - 0.5a + 0.5b^2

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
```

**IMPORTANT:** Verify the ManyWell32D energy definition against the DGFS paper before training. The sign conventions vary between papers. The energy above should satisfy: (1) E has two minima in each a-dimension, (2) E is quadratic (single minimum) in each b-dimension. Quick check:

```python
e = ManyWell32DEnergy()
x = torch.zeros(1, 32)
print(e.eval(x))  # Should be finite
# Check that E has two wells in first a-dimension
a_vals = torch.linspace(-3, 3, 100)
x_test = torch.zeros(100, 32)
x_test[:, 0] = a_vals
E_test = e.eval(x_test)
# Plot E_test vs a_vals — should show two minima
```

-----

## 2. Reference Samples

### Unequal GMM (analytical)

```python
def generate_unequal_gmm_reference(n=10000):
    """Sample from the mixture by ancestral sampling."""
    energy = UnequalGMMEnergy()
    weights = energy.get_weights()
    centers = energy.get_centers()
    std = energy.get_std()

    # Sample mode indices according to weights
    mode_idx = torch.multinomial(weights, n, replacement=True)  # (n,)
    # Sample from chosen Gaussian
    samples = centers[mode_idx] + std * torch.randn(n, 2)
    return samples

samples = generate_unequal_gmm_reference(10000)
np.save("data/test_split_UnequalGMM.npy", samples.numpy())
```

### MW5 (analytical per dimension)

```python
def generate_mw5_reference(n=10000, n_burnin=5000):
    """Each dimension is independent 1D double-well.
    Use rejection sampling or 1D MCMC per dimension, then combine."""
    import torch

    def sample_1d_double_well(n_samples, n_burnin=5000):
        """Sample from p(a) ∝ exp(-(a^4 - 6a^2 - 0.5a)) via MCMC."""
        samples = torch.zeros(n_samples)
        x = torch.tensor(0.0)
        step_size = 0.5
        total = n_samples + n_burnin
        idx = 0
        for i in range(total):
            x_new = x + step_size * torch.randn(1).item()
            E_old = x**4 - 6*x**2 - 0.5*x
            E_new = x_new**4 - 6*x_new**2 - 0.5*x_new
            if torch.rand(1).item() < min(1.0, np.exp(-(E_new - E_old))):
                x = x_new
            if i >= n_burnin:
                samples[i - n_burnin] = x
        return samples

    all_dims = []
    for d in range(5):
        s = sample_1d_double_well(n, n_burnin)
        all_dims.append(s)

    samples = torch.stack(all_dims, dim=1)  # (n, 5)
    np.save("data/test_split_MW5.npy", samples.numpy())
    return samples
```

### 32D Many-Well (analytical per pair)

```python
def generate_manywell32_reference(n=10000, n_burnin=5000):
    """Each 2D pair is independent. a-dim: MCMC. b-dim: Gaussian."""
    all_pairs = []
    for pair in range(16):
        a_samples = sample_1d_double_well(n, n_burnin)  # same as MW5
        b_samples = torch.randn(n)  # b ~ N(0, 1) since E_b = 0.5*b^2
        all_pairs.extend([a_samples, b_samples])

    samples = torch.stack(all_pairs, dim=1)  # (n, 32)
    np.save("data/test_split_ManyWell32.npy", samples.numpy())
    return samples
```

Run all three reference generation scripts before training.

-----

## 3. Config Files

### Problem Configs

`configs/problem/unequal_gmm.yaml`:

```yaml
# @package _global_
dim: 2
n_particles: 1
spatial_dim: 2

energy:
  _target_: adjoint_samplers.energies.new_benchmarks.UnequalGMMEnergy
  dim: ${dim}

evaluator:
  _target_: adjoint_samplers.components.evaluator.SyntheticEenergyEvaluator
  ref_samples_path: data/test_split_UnequalGMM.npy
```

`configs/problem/mw5.yaml`:

```yaml
# @package _global_
dim: 5
n_particles: 1
spatial_dim: 5

energy:
  _target_: adjoint_samplers.energies.new_benchmarks.ManyWell5DEnergy
  dim: ${dim}

evaluator:
  _target_: adjoint_samplers.components.evaluator.SyntheticEenergyEvaluator
  ref_samples_path: data/test_split_MW5.npy
```

`configs/problem/manywell32.yaml`:

```yaml
# @package _global_
dim: 32
n_particles: 1
spatial_dim: 32

energy:
  _target_: adjoint_samplers.energies.new_benchmarks.ManyWell32DEnergy
  dim: ${dim}

evaluator:
  _target_: adjoint_samplers.components.evaluator.SyntheticEenergyEvaluator
  ref_samples_path: data/test_split_ManyWell32.npy
```

### Experiment Configs

All three follow the same pattern. Use FourierMLP (not EGNN — these are not particle systems), standard VESDE, Gaussian source.

**Unequal GMM** — `configs/experiment/unequal_gmm_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: unequal_gmm
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: score_term_cost
  - /matcher@adjoint_matcher: adjoint_ve

exp_name: unequal_gmm_asbs
scale: 2
nfe: 100
sigma_max: 8
sigma_min: 0.01
rescale_t: null
num_epochs: 3000
max_grad_E_norm: 50

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/unequal_gmm_ksd_asbs.yaml`: Same but change matcher to SDR matcher and add `sdr_lambda: 1.0`.

**MW5** — `configs/experiment/mw5_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: mw5
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: score_term_cost
  - /matcher@adjoint_matcher: adjoint_ve

exp_name: mw5_asbs
scale: 2
nfe: 200
sigma_max: 3
sigma_min: 0.001
rescale_t: null
num_epochs: 5000
max_grad_E_norm: 100

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/mw5_ksd_asbs.yaml`: Same with SDR matcher, `sdr_lambda: 1.0`.

**32D Many-Well** — `configs/experiment/manywell32_asbs.yaml`:

```yaml
# @package _global_
defaults:
  - /problem: manywell32
  - /source: gauss
  - /sde@ref_sde: ve
  - /model@controller: fouriermlp
  - /state_cost: zero
  - /term_cost: score_term_cost
  - /matcher@adjoint_matcher: adjoint_ve

exp_name: manywell32_asbs
scale: 2
nfe: 200
sigma_max: 3
sigma_min: 0.001
rescale_t: null
num_epochs: 5000
max_grad_E_norm: 100

adjoint_matcher:
  buffer_size: 10000
  duplicates: 10
  resample_size: 1000
  num_epochs_per_stage: ${num_epochs}
  optim:
    lr: 1e-3
    weight_decay: 0

use_wandb: false
eval_freq: 200
```

`configs/experiment/manywell32_ksd_asbs.yaml`: Same with SDR matcher, `sdr_lambda: 1.0`.

**NOTE on term_cost:** These are NOT particle systems and do NOT use the corrector. Use `score_term_cost` (AS-style memoryless condition), not `graph_corrector_term_cost`. If `score_term_cost` doesn’t exist or doesn’t work, try `term_cost` with appropriate settings. The terminal cost should be $\Phi_0(x) = E(x) + \log p_1^{\text{ref}}(x)$ where $p_1^{\text{ref}}$ is the Gaussian reference marginal.

**NOTE on sigma_max:** For Unequal GMM (modes at radius 5), use sigma_max=8. For MW5 (wells at ~±1.7), use sigma_max=3. For ManyWell32 (same well structure), use sigma_max=3.

-----

## 4. Training

```bash
# Generate reference samples first
python scripts/generate_reference_samples.py  # (create this script with the code from Section 2)

# Unequal GMM (~10 min each)
python train.py experiment=unequal_gmm_asbs seed=0 use_wandb=false
python train.py experiment=unequal_gmm_ksd_asbs seed=0 use_wandb=false

# MW5 (~20 min each)
python train.py experiment=mw5_asbs seed=0 use_wandb=false
python train.py experiment=mw5_ksd_asbs seed=0 use_wandb=false

# 32D Many-Well (~30 min each)
python train.py experiment=manywell32_asbs seed=0 use_wandb=false
python train.py experiment=manywell32_ksd_asbs seed=0 use_wandb=false
```

-----

## 5. Evaluation Metrics

### 5.1 Unequal GMM (2D, 5 modes)

**Primary metrics — mode weight recovery:**

```python
def evaluate_unequal_gmm(samples, energy):
    """The key metric: does the model recover the correct mode weights?"""
    centers = energy.get_centers().to(samples.device)
    true_weights = energy.get_weights().to(samples.device)
    std = energy.get_std()
    N = samples.shape[0]

    # Assign each sample to nearest mode
    dists = torch.cdist(samples, centers)  # (N, 5)
    assignments = dists.argmin(dim=1)  # (N,)

    # Empirical weights
    empirical_weights = torch.zeros(5, device=samples.device)
    for k in range(5):
        empirical_weights[k] = (assignments == k).float().mean()

    # Metrics
    results = {
        'empirical_weights': empirical_weights.cpu().tolist(),
        'true_weights': true_weights.cpu().tolist(),

        # Total variation between empirical and true mode weights
        'weight_TV': 0.5 * (empirical_weights - true_weights.to(samples.device)).abs().sum().item(),

        # Did the 3% mode survive? (>0 samples assigned to it)
        'minority_mode_alive': (empirical_weights[4] > 0).item(),

        # Minority mode weight (should be ~0.03)
        'minority_mode_weight': empirical_weights[4].item(),

        # KL between empirical and true weights
        'weight_KL': (true_weights * torch.log(true_weights / (empirical_weights + 1e-10))).sum().item(),
    }
    return results
```

**Key result to report:** Baseline minority mode weight (should be ~0 or very small) vs SDR minority mode weight (should be ~0.03).

### 5.2 MW5 (5D, 32 modes)

**Primary metrics — mode coverage count:**

```python
def evaluate_mw5(samples, energy):
    """Count how many of 32 modes are covered."""
    centers = energy.get_mode_centers().to(samples.device)  # (32, 5)
    N = samples.shape[0]

    # Assign each sample to nearest mode center
    dists = torch.cdist(samples, centers)  # (N, 32)
    assignments = dists.argmin(dim=1)  # (N,)
    min_dists = dists.min(dim=1).values  # (N,)

    # Mode is "covered" if at least 1 sample within threshold
    threshold = 1.0  # roughly 2*std of each well
    covered_mask = torch.zeros(32, dtype=torch.bool)
    for k in range(32):
        mode_samples = (assignments == k)
        if mode_samples.any():
            if min_dists[mode_samples].min() < threshold:
                covered_mask[k] = True

    # Per-mode counts
    counts = torch.zeros(32)
    for k in range(32):
        counts[k] = (assignments == k).sum()

    results = {
        'modes_covered': covered_mask.sum().item(),
        'modes_total': 32,
        'mode_counts': counts.cpu().tolist(),
        'weight_TV': 0.5 * (counts / N - 1.0 / 32).abs().sum().item(),
        'energy_mean': energy.eval(samples).mean().item(),
    }
    return results
```

**Additionally, compute per-dimension marginal metrics:**

```python
def evaluate_mw5_marginals(samples, ref_samples):
    """For each of 5 dimensions, compute 1D Wasserstein and check bimodality."""
    from scipy.stats import wasserstein_distance
    results = {}
    for d in range(5):
        gen = samples[:, d].cpu().numpy()
        ref = ref_samples[:, d].cpu().numpy()
        w1 = wasserstein_distance(gen, ref)
        results[f'dim{d}_W1'] = w1

        # Check bimodality: fraction of samples in left well (a < 0) vs right well
        frac_left_gen = (gen < 0).mean()
        frac_left_ref = (ref < 0).mean()
        results[f'dim{d}_frac_left_gen'] = frac_left_gen
        results[f'dim{d}_frac_left_ref'] = frac_left_ref
    return results
```

### 5.3 32D Many-Well (32D, 65536 modes)

**Primary metric — log Z estimation bias (standard in PIS/DDS/DGFS literature):**

This is the metric all papers report. Use the ESS/Girsanov framework from our ESS_EVALUATION_GUIDE.md:

```python
def evaluate_manywell32_logZ(sde, source, timesteps_cfg, energy, n_samples=2000):
    """Estimate log Z using trajectory importance sampling.

    log Z = log(1/N sum_i exp(log_w_i))

    where log_w_i is the Girsanov log-weight plus terminal energy ratio.
    Compare estimated log Z to the analytical value.
    """
    # Use sdeint_with_noise and compute_girsanov_log_weights
    # from trajectory_is.py (already implemented for ESS evaluation)

    from adjoint_samplers.components.trajectory_is import (
        sdeint_with_noise, compute_girsanov_log_weights
    )

    states, noises, dts = sdeint_with_noise(sde, source.sample([n_samples]).cuda(), timesteps)
    log_w = compute_girsanov_log_weights(sde, states, noises, dts, timesteps)

    x1 = states[-1]
    x0 = states[0]

    # log importance weight for Z estimation
    log_unnorm = -energy.eval(x1) - source.log_prob(x0) + log_w

    # log Z estimate (IWAE-style lower bound)
    log_Z_est = torch.logsumexp(log_unnorm, dim=0) - np.log(n_samples)

    # Analytical log Z for ManyWell32:
    # Z = (Z_pair)^16 where Z_pair = Z_a * Z_b
    # Z_b = sqrt(2*pi) (Gaussian integral)
    # Z_a = integral exp(-(a^4 - 6a^2 - 0.5a)) da  (compute numerically)
    a = torch.linspace(-10, 10, 100000)
    da = a[1] - a[0]
    log_Za = torch.logsumexp(-(a**4 - 6*a**2 - 0.5*a), dim=0) + torch.log(da)
    log_Zb = 0.5 * np.log(2 * np.pi)
    log_Z_true = 16 * (log_Za.item() + log_Zb)

    bias = abs(log_Z_est.item() - log_Z_true)

    return {
        'log_Z_est': log_Z_est.item(),
        'log_Z_true': log_Z_true,
        'log_Z_bias': bias,
    }
```

**Also compute per-pair marginals (the standard visualization from DGFS Figure 4):**

```python
def evaluate_manywell32_marginals(samples, ref_samples):
    """For each of 16 pairs, compute marginal metrics on the a-dimension."""
    from scipy.stats import wasserstein_distance
    results = {}
    for pair in range(16):
        a_idx = 2 * pair
        gen_a = samples[:, a_idx].cpu().numpy()
        ref_a = ref_samples[:, a_idx].cpu().numpy()

        w1 = wasserstein_distance(gen_a, ref_a)
        frac_left_gen = (gen_a < 0).mean()
        frac_left_ref = (ref_a < 0).mean()

        results[f'pair{pair}_a_W1'] = w1
        results[f'pair{pair}_frac_left_gen'] = frac_left_gen
        results[f'pair{pair}_frac_left_ref'] = frac_left_ref

    # Average across pairs
    results['mean_a_W1'] = np.mean([results[f'pair{p}_a_W1'] for p in range(16)])
    return results
```

-----

## 6. Visualizations

### 6.1 Unequal GMM — Mode Weight Bar Chart (KEY FIGURE)

```python
def plot_unequal_gmm_weights(results_base, results_ksd, output_path):
    """Bar chart: true weights vs baseline vs SDR-ASBS per mode.

    This is THE figure for this benchmark. It should show:
    - Mode 5 (3%) has ~0% weight under baseline, ~3% under SDR
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(5)
    width = 0.25

    true_w = results_base['true_weights']
    base_w = results_base['empirical_weights']
    ksd_w = results_ksd['empirical_weights']

    ax.bar(x - width, true_w, width, label='Target', color='gray', alpha=0.7)
    ax.bar(x, base_w, width, label='ASBS', color='#d62728', alpha=0.7)
    ax.bar(x + width, ksd_w, width, label='SDR-ASBS', color='#ff7f0e', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Mode {i+1}\n($w$={true_w[i]:.2f})' for i in range(5)])
    ax.set_ylabel('Empirical Weight')
    ax.set_title('Mode Weight Recovery: Unequal GMM')
    ax.legend()

    # Annotate the minority mode
    ax.annotate('Minority mode\n(target: 3%)',
                xy=(4, max(base_w[4], ksd_w[4]) + 0.01),
                ha='center', fontsize=9, color='red')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
```

Also generate the standard terminal distribution + trajectory plots using the same eval_2d_viz.py framework from the 2D benchmark guide.

### 6.2 MW5 — Per-Dimension Marginal Histograms (5-panel)

```python
def plot_mw5_marginals(samples_base, samples_ksd, ref_samples, output_path):
    """5-panel figure: one per dimension.
    Each panel shows overlaid histograms of reference, baseline, SDR.
    Key visual: baseline may collapse to one well in some dimensions,
    SDR should have both wells populated in all dimensions.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for d in range(5):
        ax = axes[d]
        ref = ref_samples[:, d].cpu().numpy()
        base = samples_base[:, d].cpu().numpy()
        ksd = samples_ksd[:, d].cpu().numpy()

        bins = np.linspace(-3.5, 3.5, 60)
        ax.hist(ref, bins=bins, density=True, alpha=0.3, color='gray', label='Reference')
        ax.hist(base, bins=bins, density=True, alpha=0.5, color='#d62728', label='ASBS')
        ax.hist(ksd, bins=bins, density=True, alpha=0.5, color='#ff7f0e', label='SDR-ASBS')

        ax.set_title(f'$x_{d+1}$')
        if d == 0:
            ax.legend(fontsize=8)
        ax.set_xlim(-3.5, 3.5)

    fig.suptitle('MW5: Per-Dimension Marginals', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
```

### 6.3 32D Many-Well — Pair Marginal Grid (KEY FIGURE for literature comparison)

```python
def plot_manywell32_marginals(samples_base, samples_ksd, ref_samples, output_path):
    """Show the (a, b) joint distribution for selected pairs.

    Standard visualization from DGFS (Figure 4): project onto (dim 0, dim 2)
    i.e., the a-dimensions of pair 0 and pair 1.

    3-panel: Reference, Baseline, SDR-ASBS.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dim_a1, dim_a2 = 0, 2  # a-dimensions of pair 0 and pair 1

    for ax, (samples, title) in zip(axes, [
        (ref_samples, 'Reference'),
        (samples_base, 'ASBS'),
        (samples_ksd, 'SDR-ASBS'),
    ]):
        x = samples[:, dim_a1].cpu().numpy()
        y = samples[:, dim_a2].cpu().numpy()
        ax.hist2d(x, y, bins=80, range=[[-3.5, 3.5], [-3.5, 3.5]],
                  cmap='hot_r', density=True)
        ax.set_xlabel(f'$x_1$ (pair 0)')
        ax.set_ylabel(f'$x_3$ (pair 1)')
        ax.set_title(title)
        ax.set_aspect('equal')

    fig.suptitle('32D Many-Well: Joint Marginal ($x_1$, $x_3$)', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
```

**This figure should show 4 clusters** (2 wells × 2 wells = 4 combinations). If baseline shows only 2 clusters (one dimension collapsed to single well), SDR shows all 4 — this is the visual proof.

### 6.4 32D Many-Well — log Z Bias Comparison Table

This is the standard result format from PIS/DDS/DGFS papers:

```
| Method     | |log Z_est - log Z_true| |
|------------|-------------------------|
| PIS†       | 1.391 ± 1.014           |
| DDS†       | 1.154 ± 0.626           |
| DGFS†      | 0.904 ± 0.067           |
| ASBS       | ?.??? ± ?.???           |
| SDR-ASBS   | ?.??? ± ?.???           |
```

†Values from DGFS Table 1.

-----

## 7. Results Formatting

After evaluation, append all results to RESULTS.md:

```markdown
### Unequal-Weight GMM (2D, 5 modes)

| Metric | ASBS | SDR-ASBS |
|---|---|---|
| Mode weight TV ↓ | | |
| Minority mode weight (target: 0.03) | | |
| Minority mode alive? | | |
| Weight KL ↓ | | |

**Mode Weight Recovery:**
![unequal_gmm_weights](figures/unequal_gmm_weights.png)

### Many-Well MW5 (5D, 32 modes)

| Metric | ASBS | SDR-ASBS |
|---|---|---|
| Modes covered (of 32) | | |
| Mode weight TV ↓ | | |
| Mean 1D Wasserstein ↓ | | |

**Per-Dimension Marginals:**
![mw5_marginals](figures/mw5_marginals.png)

### 32D Many-Well (32D, 65536 modes)

| Metric | PIS† | DDS† | DGFS† | ASBS | SDR-ASBS |
|---|---|---|---|---|---|
| |log Z bias| ↓ | 1.391 | 1.154 | 0.904 | | |
| Mean pair W1 ↓ | | | | | |

**Joint Marginal ($x_1$, $x_3$):**
![manywell32_marginals](figures/manywell32_marginals.png)
```

-----

## 8. Important Notes

1. **These are NOT particle systems.** Use `FourierMLP`, `VESDE` (not `GraphVESDE`), `gauss` source (not `harmonic`). Do NOT use EGNN. Do NOT use graph-based configs.
1. **Term cost:** Use AS-style (memoryless, no corrector) since these are non-molecular with Gaussian source. The terminal cost should be $\Phi_0(x) = E(x) + \log p_1^{\text{ref}}(x)$ where $p_1^{\text{ref}}$ is the Gaussian reference marginal. Check which `term_cost` config achieves this — likely `score_term_cost`. If training diverges, try adding the corrector (ASBS-style with `corrector_term_cost`).
1. **Verify the 32D energy definition.** The sign convention on the Many-Well potential varies between papers. Cross-check against DGFS (Zhang et al., ICLR 2024) Section 5.1 and PIS (Zhang & Chen, 2022). The key property: each 2D pair should have exactly 2 energy minima when projected onto the a-axis.
1. **sigma_max matters.** For Unequal GMM (modes at radius 5): sigma_max=8. For MW5 and ManyWell32 (wells at ~±1.7): sigma_max=3. Too small: SDE can’t reach the modes. Too large: training is unstable.
1. **SDR lambda may need tuning.** Start with lambda=1.0 for all three. If MW5 or ManyWell32 diverge, try lambda=0.5 or 0.1. If Unequal GMM is too weak, try lambda=2.0.
1. **32D Many-Well log Z requires trajectory IS infrastructure.** Use the `sdeint_with_noise` and `compute_girsanov_log_weights` functions from `ESS_EVALUATION_GUIDE.md`. If these haven’t been implemented yet, implement them first.
1. **Run 5 eval seeds** (0-4) with 2000 samples each, same as all other benchmarks. Report mean ± std.