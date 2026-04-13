# ESS Evaluation via Girsanov Importance Weights

## Guide for Claude Code — Add to Existing Evaluation Pipeline

-----

## 1. What We’re Computing and Why

We want to show that SDR-ASBS produces trajectories with **higher Effective Sample Size (ESS)** for importance sampling than baseline ASBS. Higher ESS means the model’s path measure better covers the target, enabling downstream tasks like partition function estimation.

We do NOT estimate $Z$ itself. We compute ESS as a **diagnostic metric** that measures how useful the generated trajectories would be for IS-based estimation.

-----

## 2. Mathematical Setup

### 2.1 The Forward SDE (What We Simulate)

For VE-SDE with controller $u_\theta$, the Euler-Maruyama discretization at step $k$ is:

$$X_{k+1} = X_k + g(t_k)^2 u_\theta(X_k, t_k)\Delta t_k + g(t_k)\sqrt{\Delta t_k},\epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I_d)$$

where:

- $g(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^{1-t}\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$ (from `VESDE.diff`)
- $\Delta t_k = t_{k+1} - t_k$
- $\epsilon_k \in \mathbb{R}^d$ is the noise realization at step $k$

The **controlled** forward path measure $Q_\theta$ has transition density:

$$q_\theta(x_{k+1}|x_k) = \mathcal{N}\left(x_{k+1};; x_k + g_k^2 u_\theta(x_k, t_k)\Delta t_k,; g_k^2\Delta t_k,I_d\right)$$

### 2.2 The Reference (Uncontrolled) Path Measure

The **uncontrolled** VE-SDE (with $u = 0$) has transition:

$$p_{\text{ref}}(x_{k+1}|x_k) = \mathcal{N}\left(x_{k+1};; x_k,; g_k^2\Delta t_k,I_d\right)$$

This is just Brownian motion with time-varying diffusion — no drift.

### 2.3 The Girsanov Log-Weight

The log-importance-weight for a single trajectory $\mathbf{X} = (X_0, X_1, \ldots, X_K)$ is the log-ratio of path measures:

$$\log W(\mathbf{X}) = \log\frac{dP_{\text{ref}}}{dQ_\theta}(\mathbf{X}) = \sum_{k=0}^{K-1} \log\frac{p_{\text{ref}}(X_{k+1}|X_k)}{q_\theta(X_{k+1}|X_k)}$$

Since both are Gaussians with the **same covariance** $g_k^2\Delta t_k,I_d$ but different means, the log-ratio simplifies. For two Gaussians $\mathcal{N}(x; \mu_1, \Sigma)$ and $\mathcal{N}(x; \mu_2, \Sigma)$:

$$\log\frac{\mathcal{N}(x;\mu_1,\Sigma)}{\mathcal{N}(x;\mu_2,\Sigma)} = -\frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1) + \frac{1}{2}(x-\mu_2)^T\Sigma^{-1}(x-\mu_2)$$

With $\mu_{\text{ref}} = x_k$, $\mu_\theta = x_k + g_k^2 u_\theta \Delta t_k$, $\Sigma = g_k^2\Delta t_k,I_d$, and $x = x_{k+1}$:

$$\delta\mu_k = \mu_\theta - \mu_{\text{ref}} = g_k^2 u_\theta(x_k, t_k)\Delta t_k$$

$$\log\frac{p_{\text{ref}}(x_{k+1}|x_k)}{q_\theta(x_{k+1}|x_k)} = \frac{1}{g_k^2\Delta t_k}\left[\delta\mu_k^T(x_{k+1} - x_k) - \frac{1}{2}|\delta\mu_k|^2\right] - \frac{1}{g_k^2\Delta t_k}\cdot 0$$

Wait — let me be more careful. Let $r = x_{k+1} - x_k$ (the increment):

$$\log\frac{p_{\text{ref}}}{q_\theta} = -\frac{|r|^2}{2g_k^2\Delta t_k} + \frac{|r - g_k^2 u_\theta\Delta t_k|^2}{2g_k^2\Delta t_k}$$

$$= \frac{1}{2g_k^2\Delta t_k}\left[|r - g_k^2 u_\theta\Delta t_k|^2 - |r|^2\right]$$

$$= \frac{1}{2g_k^2\Delta t_k}\left[-2r^T g_k^2 u_\theta\Delta t_k + g_k^4|u_\theta|^2\Delta t_k^2\right]$$

$$= -u_\theta(x_k, t_k)^T r + \frac{1}{2}g_k^2|u_\theta(x_k, t_k)|^2\Delta t_k$$

Now substitute $r = x_{k+1} - x_k = g_k^2 u_\theta\Delta t_k + g_k\sqrt{\Delta t_k}\epsilon_k$:

$$= -u_\theta^T\left[g_k^2 u_\theta\Delta t_k + g_k\sqrt{\Delta t_k}\epsilon_k\right] + \frac{1}{2}g_k^2|u_\theta|^2\Delta t_k$$

$$= -g_k^2|u_\theta|^2\Delta t_k - g_k\sqrt{\Delta t_k},u_\theta^T\epsilon_k + \frac{1}{2}g_k^2|u_\theta|^2\Delta t_k$$

$$\boxed{\log\frac{p_{\text{ref}}(x_{k+1}|x_k)}{q_\theta(x_{k+1}|x_k)} = -g_k\sqrt{\Delta t_k},u_\theta(x_k, t_k)^T\epsilon_k - \frac{1}{2}g_k^2|u_\theta(x_k, t_k)|^2\Delta t_k}$$

This is the **discrete Girsanov formula**. The full trajectory log-weight is:

$$\boxed{\log W(\mathbf{X}) = -\sum_{k=0}^{K-1}\left[g_k\sqrt{\Delta t_k},u_\theta(x_k, t_k)^T\epsilon_k + \frac{1}{2}g_k^2|u_\theta(x_k, t_k)|^2\Delta t_k\right]}$$

### 2.4 The Full Importance Weight for Z Estimation

For partition function estimation, we need:

$$\hat{Z} = \frac{1}{N}\sum_{i=1}^N \tilde{w}_i, \quad \tilde{w}*i = \frac{\exp(-E(X_1^i))}{q*\theta(X_1^i)}$$

Using the Girsanov weight and the source density $\mu(X_0)$:

$$\log\tilde{w}_i = -E(X_1^i) - \log\mu(X_0^i) + \log W(\mathbf{X}^i)$$

But we don’t need $\tilde{w}_i$ for ESS. We only need the **self-normalized** importance weights:

$$w_i = \frac{\exp(\log W(\mathbf{X}^i))}{\sum_j \exp(\log W(\mathbf{X}^j))}$$

### 2.5 Effective Sample Size

$$\text{ESS} = \frac{1}{\sum_{i=1}^N w_i^2}$$

where $w_i$ are the self-normalized weights. ESS ranges from 1 (all weight on one sample) to $N$ (uniform weights). Higher ESS means the controlled SDE’s path measure is closer to the reference — i.e., the controller is more “efficient” at steering toward the target.

**For SDR-ASBS vs baseline:** If SDR-ASBS covers more modes, its trajectory weights are more uniform (no single trajectory gets all the weight), giving higher ESS.

-----

## 3. Implementation

### 3.1 Modified sdeint That Returns Noise

Create a new function `sdeint_with_noise` — do NOT modify the original `sdeint`.

```python
"""
Add to adjoint_samplers/components/sde.py (or a new file trajectory_is.py)
"""

@torch.no_grad()
def sdeint_with_noise(
    sde: BaseSDE,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Forward SDE integration that also returns noise realizations.

    Returns:
        states: list of (B, D) tensors — trajectory [x_0, x_1, ..., x_K]
        noises: list of (B, D) tensors — noise realizations [ε_0, ε_1, ..., ε_{K-1}]
        dts: list of scalar tensors — time steps [Δt_0, Δt_1, ..., Δt_{K-1}]
    """
    T = len(timesteps)
    state = state0.clone()
    states = [state0]
    noises = []
    dts = []

    for i in range(T - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        # Generate noise BEFORE computing diffusion
        eps = sde.randn_like(state)  # ε_k ~ N(0, I)

        drift = sde.drift(t, state) * dt
        diffusion = sde.diff(t) * dt.sqrt() * eps

        d_state = drift + diffusion
        state = sde.propagate(state, d_state)

        states.append(state)
        noises.append(eps)
        dts.append(dt)

    return states, noises, dts
```

### 3.2 Girsanov Log-Weight Computation

```python
@torch.no_grad()
def compute_girsanov_log_weights(
    sde: ControlledSDE,
    states: list[torch.Tensor],
    noises: list[torch.Tensor],
    dts: list[torch.Tensor],
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Compute log-importance-weight for each trajectory via Girsanov formula.

    log W = -Σ_k [g_k √Δt_k u_θ(x_k, t_k)^T ε_k + ½ g_k² ||u_θ(x_k, t_k)||² Δt_k]

    Args:
        sde: ControlledSDE (has sde.u = controller, sde.ref_sde = reference)
        states: list of K+1 tensors, each (B, D)
        noises: list of K tensors, each (B, D) — the ε_k used during simulation
        dts: list of K scalar tensors
        timesteps: (K+1,) tensor

    Returns:
        log_w: (B,) tensor — log-importance-weight per trajectory
    """
    B = states[0].shape[0]
    log_w = torch.zeros(B, device=states[0].device)

    K = len(noises)
    for k in range(K):
        t_k = timesteps[k]
        x_k = states[k]
        eps_k = noises[k]
        dt_k = dts[k]

        # Controller output
        u_k = sde.u(t_k, x_k)  # (B, D)

        # Diffusion coefficient
        g_k = sde.ref_sde.diff(t_k)  # scalar or (B,) or (B,1)

        # Ensure g_k is broadcastable
        if g_k.dim() == 0:
            g_k = g_k.unsqueeze(0)
        while g_k.dim() < u_k.dim():
            g_k = g_k.unsqueeze(-1)

        # Stochastic integral term: g_k * sqrt(dt) * u^T ε
        stoch_term = (g_k * dt_k.sqrt() * u_k * eps_k).sum(dim=-1)  # (B,)

        # Control cost term: ½ g_k² ||u||² dt
        cost_term = 0.5 * (g_k**2 * (u_k**2).sum(dim=-1, keepdim=True) * dt_k).squeeze(-1)  # (B,)

        log_w = log_w - stoch_term - cost_term

    return log_w
```

### 3.3 ESS Computation

```python
def compute_ess(log_weights: torch.Tensor) -> float:
    """Compute Effective Sample Size from log-importance-weights.

    ESS = (Σ w_i)² / Σ w_i²

    Using log-sum-exp for numerical stability.

    Args:
        log_weights: (N,) tensor of log W_i

    Returns:
        ess: scalar, in range [1, N]
    """
    N = log_weights.shape[0]

    # Normalize: log w_i - log(Σ exp(log w_j))
    log_w_normalized = log_weights - torch.logsumexp(log_weights, dim=0)

    # ESS = 1 / Σ w_i²  where w_i are normalized (sum to 1)
    # log(Σ w_i²) = log(Σ exp(2 log w_i)) = logsumexp(2 * log_w_normalized)
    log_sum_sq = torch.logsumexp(2 * log_w_normalized, dim=0)
    ess = torch.exp(-log_sum_sq).item()

    return ess
```

### 3.4 Full Evaluation Function

```python
def evaluate_ess(
    sde: ControlledSDE,
    source,
    timesteps_cfg: dict,
    n_samples: int = 2000,
    batch_size: int = 500,
    device: str = 'cuda',
) -> dict:
    """Evaluate ESS for a trained model.

    Args:
        sde: trained ControlledSDE
        source: source distribution (Gaussian / harmonic)
        timesteps_cfg: dict for train_utils.get_timesteps
        n_samples: total trajectories to evaluate
        batch_size: trajectories per batch
        device: torch device

    Returns:
        dict with 'ess', 'ess_fraction', 'log_weights_mean', 'log_weights_std',
             'control_cost_mean'
    """
    import adjoint_samplers.utils.train_utils as train_utils

    all_log_weights = []
    all_control_costs = []

    sde.eval()
    n_generated = 0

    while n_generated < n_samples:
        b = min(batch_size, n_samples - n_generated)

        x0 = source.sample([b]).to(device)
        timesteps = train_utils.get_timesteps(**timesteps_cfg).to(device)

        # Forward SDE with noise tracking
        states, noises, dts = sdeint_with_noise(sde, x0, timesteps)

        # Girsanov log-weights
        log_w = compute_girsanov_log_weights(sde, states, noises, dts, timesteps)
        all_log_weights.append(log_w)

        # Also compute total control cost: Σ_k ½ g_k² ||u_k||² Δt_k
        control_cost = torch.zeros(b, device=device)
        for k in range(len(noises)):
            t_k = timesteps[k]
            u_k = sde.u(t_k, states[k])
            g_k = sde.ref_sde.diff(t_k)
            dt_k = dts[k]
            if g_k.dim() == 0:
                g_k = g_k.unsqueeze(0)
            control_cost += 0.5 * (g_k**2 * (u_k**2).sum(dim=-1) * dt_k).squeeze()
        all_control_costs.append(control_cost)

        n_generated += b

    # Concatenate
    log_weights = torch.cat(all_log_weights)[:n_samples]
    control_costs = torch.cat(all_control_costs)[:n_samples]

    # ESS
    ess = compute_ess(log_weights)

    return {
        'ess': ess,
        'ess_fraction': ess / n_samples,
        'n_samples': n_samples,
        'log_weights_mean': log_weights.mean().item(),
        'log_weights_std': log_weights.std().item(),
        'log_weights_min': log_weights.min().item(),
        'log_weights_max': log_weights.max().item(),
        'control_cost_mean': control_costs.mean().item(),
        'control_cost_std': control_costs.std().item(),
    }
```

-----

## 4. Integration with Existing Evaluation

### 4.1 Where to Add

In `evaluate_all.py`, after computing all existing metrics for each experiment, add:

```python
# --- ESS evaluation ---
from adjoint_samplers.components.trajectory_is import evaluate_ess

ess_results = evaluate_ess(
    sde=exp['sde'],
    source=exp['source'],
    timesteps_cfg=exp['ts_cfg'],
    n_samples=2000,
    batch_size=500,
    device=device,
)
for k, v in ess_results.items():
    metrics[f'ess_{k}'] = v
```

### 4.2 Results Table to Add to RESULTS.md

```markdown
## Trajectory Importance Sampling: Effective Sample Size

ESS measures how well the controlled SDE's path measure covers the target.
Higher ESS → more uniform importance weights → better partition function estimation.

| Benchmark | Method | ESS | ESS/N (%) | log W (mean±std) | Control Cost |
|---|---|---|---|---|---|
| DW4 | Baseline ASBS | | | | |
| DW4 | SDR-ASBS (λ=1.0) | | | | |
| LJ13 | Baseline ASBS | | | | |
| LJ13 | SDR-ASBS (λ=1.0) | | | | |
| RotGMM d=10 | Baseline ASBS | | | | |
| RotGMM d=10 | SDR-ASBS (λ=1.0) | | | | |
```

### 4.3 What to Expect

- **ESS (baseline):** Low, especially on multimodal targets. If the sampler mode-collapses, a few trajectories that accidentally reach underrepresented modes get enormous weight, collapsing ESS.
- **ESS (SDR-ASBS):** Higher, because SDR forces mode coverage. More modes covered → weights more uniform → higher ESS.
- **log W standard deviation:** This is the key diagnostic. If `std(log W)` is large (say >10), ESS will be near 1 regardless of N. SDR should reduce this variance.
- **Control cost:** Should be similar between baseline and SDR (the SDR correction modifies the adjoint target, not the control cost directly). If SDR’s control cost is much higher, the controller is working harder to spread particles.

-----

## 5. Important Notes

1. **`sdeint_with_noise` must use the SAME noise generation as `sdeint`.** The noise is drawn via `sde.randn_like(state)`, which for `GraphVESDE` generates COM-free noise. The new function must use the same method — do NOT use `torch.randn_like` directly.
1. **Controller call convention.** The EGNN controller expects `(t, x)` not `(x, t)`. Check the signature: `sde.u(t, x)` where `t` is a scalar tensor and `x` is `(B, D)`.
1. **No gradients needed.** The entire ESS computation runs under `@torch.no_grad()`. We evaluate the controller $u_\theta(x_k, t_k)$ at each step but don’t backprop.
1. **Memory:** We need to store the controller output at every step (for the log-weight), not just the boundary. For $K = 200$ steps, $B = 500$ batch, $D = 8$ (DW4): 200 × 500 × 8 × 4 bytes = 3.2 MB — trivial. For LJ55 ($D = 165$): 200 × 500 × 165 × 4 = 66 MB — fine.
1. **Numerical stability.** The log-weights can span a huge range (hundreds in magnitude). The `compute_ess` function uses `logsumexp` throughout to avoid overflow. Never exponentiate log-weights directly.
1. **Batch processing.** Process trajectories in batches of 500 to avoid OOM from storing all intermediate states. The `evaluate_ess` function handles this.
1. **The Girsanov formula assumes we know the exact noise $\epsilon_k$ used.** This is why we need `sdeint_with_noise` — the standard `sdeint` generates and discards the noise.