# SDR-DARW Implementation Guide for Claude Code

## Overview

Add **Density-Adaptive Regression Reweighting (SDR-DARW)** to the existing SDR-ASBS codebase. SDR-DARW reweights the Adjoint Matching loss per-sample using importance weights derived from the same kernel matrix already computed for SDR. The goal is to amplify the loss contribution of particles in under-represented modes and attenuate over-represented ones.

**Paper equation (reweighted AM loss):**
```
L_AM^rw = (1/N) Σ_i ŵ_i · ‖u_θ(x_t^i, t) + g²(a_i + Δ_i)‖²
```
where `ŵ_i = (exp(-E(x1_i)) / q̂(x1_i))^β / [(1/N) Σ_m (exp(-E(x1_m)) / q̂(x1_m))^β]` and `q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j)`.

## Architecture Summary

The relevant data flow is:

```
populate_buffer (sdr_matcher.py)
  → forward SDE → get x0, x1
  → compute adjoint1 = ∇E(x1) - h_ψ(x1)
  → compute SDR correction Δ_i, add to adjoint1
  → store (x0, x1, adjoint1) in buffer           ← SDR-DARW weights computed here

prepare_target (matcher.py)
  → read from buffer
  → sample t, compute x_t
  → return (t, x_t), -adjoint1                   ← also return weights

train_one_epoch (train_loop.py)
  → input, target = matcher.prepare_target(data)
  → output = model(*input)
  → loss = ((output - target)**2).mean()          ← apply weights here
```

## Implementation Steps

### Step 1: Add SDR-DARW weight computation to `sdr_matcher.py`

In `SDRAdjointVEMatcher`, add a method `_compute_darw_weights` and call it from `populate_buffer`. The weights should be computed from the **same x1 batch** and reuse the base kernel evaluations.

**Add these constructor parameters to `__init__`:**
```python
sdr_beta: float = 0.0,        # 0 = disabled (uniform weights), 1 = full reweighting
darw_weight_clip: float = 10.0, # max allowed weight (for stability)
```

When `sdr_beta == 0`, SDR-DARW is disabled and all weights are 1.0.

**Add this method:**
```python
@torch.no_grad()
def _compute_darw_weights(self, x1: torch.Tensor) -> torch.Tensor:
    """Compute SDR-DARW importance weights for the terminal batch.
    
    ŵ_i = (exp(-E(x1_i)) / q̂(x1_i))^β, self-normalized
    
    where q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j) is the KDE using the
    base kernel (same bandwidth as SDR).
    """
    N, D = x1.shape
    device = x1.device
    
    # 1. Compute base kernel matrix k(x1_i, x1_j)
    #    Use the same bandwidth as SDR (median heuristic or fixed)
    if self.ksd_bandwidth is not None:
        ell = torch.tensor(self.ksd_bandwidth, device=device, dtype=x1.dtype)
    else:
        ell = median_bandwidth(x1)
    
    # Pairwise squared distances
    diffs = x1.unsqueeze(0) - x1.unsqueeze(1)  # (N, N, D)
    sq_dists = (diffs ** 2).sum(dim=-1)          # (N, N)
    
    # RBF kernel matrix
    K = torch.exp(-sq_dists / (2 * ell ** 2))    # (N, N)
    
    # 2. KDE: q̂(x1_i) = (1/N) Σ_j k(x1_i, x1_j)
    q_hat = K.mean(dim=1)                         # (N,)
    q_hat = q_hat.clamp(min=1e-10)                # avoid division by zero
    
    # 3. Unnormalized target: p̃(x1_i) = exp(-E(x1_i))
    #    We need E(x1_i). Get it from the energy function.
    with torch.enable_grad():
        x1_req = x1.clone().detach().requires_grad_(True)
        energy_out = self.grad_term_cost.energy(x1_req)
    energies = energy_out["energy"].detach()       # (N,)
    
    # Stabilize: shift energies by min to avoid exp underflow
    energies = energies - energies.min()
    p_tilde = torch.exp(-energies)                 # (N,)
    
    # 4. Raw importance ratios
    ratios = p_tilde / q_hat                       # (N,)
    
    # 5. Soft power clipping with β
    ratios_beta = ratios ** self.sdr_beta          # (N,)
    
    # 6. Hard clip for stability
    ratios_beta = ratios_beta.clamp(max=self.darw_weight_clip)
    
    # 7. Self-normalize so mean = 1
    weights = ratios_beta / ratios_beta.mean()     # (N,), mean = 1
    
    return weights
```

**IMPORTANT NOTES:**
- For large N, the `(N, N, D)` tensor for `diffs` may OOM. Add chunked computation if `N > ksd_efficient_threshold`, similar to the existing `compute_stein_kernel_gradient_efficient`. The simplest approach: compute `K.mean(dim=1)` row by row in chunks.
- The energy call `self.grad_term_cost.energy(x1_req)` already exists in `_apply_ksd_correction` for computing scores. You may want to cache the energy values to avoid redundant computation. Or compute both SDR and SDR-DARW in a single pass.
- The `energy_out["energy"]` field must exist for all energy functions. Check each energy class in `adjoint_samplers/energies/` to verify they return an `"energy"` key. If not, add it. Most return `"forces"` (the gradient); you may need to also return the scalar energy.

### Step 2: Modify `populate_buffer` in `SDRAdjointVEMatcher`

Change `populate_buffer` to compute and store SDR-DARW weights:

```python
def populate_buffer(self, x0, timesteps, is_asbs_init_stage, epoch=-1):
    # Step 1: Forward SDE (unchanged)
    (x0, x1) = sdeint(self.sde, x0, timesteps, only_boundary=True)

    # Step 2: Standard adjoint (unchanged)
    adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

    # Step 3: SDR correction (unchanged)
    in_warmup = (self.ksd_warmup_epochs > 0 and epoch >= 0
                 and epoch < self.ksd_warmup_epochs)
    if self.sdr_lambda > 0 and not is_asbs_init_stage and not in_warmup:
        adjoint1 = self._apply_ksd_correction(x1, adjoint1)

    # Step 4: SDR-DARW weights (NEW)
    if self.sdr_beta > 0 and not is_asbs_init_stage and not in_warmup:
        darw_weights = self._compute_darw_weights(x1)
    else:
        darw_weights = torch.ones(x1.shape[0], device=x1.device)

    # Step 5: Store in buffer (MODIFIED — add weights)
    self._check_buffer_sample_shape(x0, x1, adjoint1)
    self.buffer.add({
        "x0": x0.to("cpu"),
        "x1": x1.to("cpu"),
        "adjoint1": adjoint1.to("cpu"),
        "darw_weights": darw_weights.to("cpu"),  # (N,)
    })
```

Do the same for `SDRAdjointVPMatcher`.

### Step 3: Modify `prepare_target` in `AdjointVEMatcher`

The base `prepare_target` in `matcher.py` needs to pass weights through. The cleanest approach: **override `prepare_target` in `SDRAdjointVEMatcher`**.

```python
def prepare_target(self, data, device):
    x0 = data["x0"].to(device)
    x1 = data["x1"].to(device)
    adjoint1 = data["adjoint1"].to(device)

    t = self.sample_t(x0).to(device)
    xt = self.sde.sample_base_posterior(t, x0, x1)
    adjoint = adjoint1

    self._check_target_shape(t, xt, adjoint)

    # Return weights if present, else uniform
    if "darw_weights" in data:
        weights = data["darw_weights"].to(device)  # (B,)
    else:
        weights = torch.ones(x0.shape[0], device=device)

    return (t, xt), -adjoint, weights
```

**IMPORTANT:** This changes the return signature from `(input, target)` to `(input, target, weights)`. You need to handle this in `train_loop.py`. Two options:

**Option A (recommended):** Make the return a dict or check length:
```python
result = matcher.prepare_target(data, device)
if len(result) == 3:
    input, target, weights = result
else:
    input, target = result
    weights = None
```

**Option B:** Always return 3 values from all matchers (add `weights = torch.ones(...)` to the base class). This is cleaner but touches more files.

### Step 4: Modify `train_one_epoch` in `train_loop.py`

Change the loss computation from unweighted MSE to weighted MSE:

**Current (line 67):**
```python
loss = loss_scale * ((output - target)**2).mean()
```

**New:**
```python
per_sample_loss = ((output - target) ** 2).sum(dim=-1)  # (B,)
if weights is not None:
    loss = loss_scale * (weights * per_sample_loss).mean()
else:
    loss = loss_scale * per_sample_loss.mean()
```

The `sum(dim=-1)` reduces over the D dimension to get a per-sample scalar loss, then `weights` (shape `(B,)`) scales each sample, then `.mean()` averages over the batch. Since weights are self-normalized (mean ≈ 1), the loss scale is preserved.

### Step 5: Add config files

**Create `configs/matcher/ksd_darw_adjoint_ve.yaml`:**
```yaml
_target_: adjoint_samplers.components.sdr_matcher.SDRAdjointVEMatcher

sdr_lambda: ${sdr_lambda}
ksd_bandwidth: null
ksd_max_particles: 2048
ksd_efficient_threshold: 1024
ksd_score_beta: 1.0
ksd_warmup_epochs: 0

sdr_beta: ${sdr_beta}
darw_weight_clip: 10.0

grad_state_cost:
  _target_: adjoint_samplers.components.state_cost.ZeroGradStateCost

buffer:
  _target_: adjoint_samplers.components.buffer.BatchBuffer
  buffer_size: ${adjoint_matcher.buffer_size}
```

**Create experiment configs** that set `sdr_beta`. For example, in the experiment YAML:
```yaml
sdr_lambda: 1.0
sdr_beta: 0.5   # moderate reweighting
```

### Step 6: Logging

Add SDR-DARW-related logging to the training loop. In `sdr_matcher.py`, add fields:
```python
self._last_darw_weight_max = 0.0
self._last_darw_weight_min = 0.0
self._last_darw_weight_std = 0.0
```

Set them in `_compute_darw_weights`:
```python
self._last_darw_weight_max = weights.max().item()
self._last_darw_weight_min = weights.min().item()
self._last_darw_weight_std = weights.std().item()
```

Log these in `train.py` alongside the existing SDR logging (search for `_last_ksd` to find where).

### Step 7: Apply the same changes to `SDRAdjointVPMatcher`

Mirror all changes for the VP variant.

## Energy Function Check

The `_compute_darw_weights` method needs `energy_out["energy"]` (the scalar energy value). Verify that every energy class returns this key:

- Check all files in `adjoint_samplers/energies/`
- The `energy()` method should return a dict with at least `{"energy": ..., "forces": ...}`
- If any energy class only returns `"forces"`, add the scalar energy computation

## Memory Considerations

For large batches (N > 1024), the `(N, N)` kernel matrix may be large. The SDR computation already handles this via chunking (`compute_stein_kernel_gradient_efficient`). For SDR-DARW, you only need `K.mean(dim=1)` (the row sums), which can be computed in chunks:

```python
def _compute_kde_chunked(self, x1, ell, chunk_size=256):
    N = x1.shape[0]
    q_hat = torch.zeros(N, device=x1.device)
    for i in range(0, N, chunk_size):
        x_chunk = x1[i:i+chunk_size]            # (C, D)
        sq_dists = torch.cdist(x_chunk, x1) ** 2  # (C, N)
        K_chunk = torch.exp(-sq_dists / (2 * ell ** 2))
        q_hat[i:i+chunk_size] = K_chunk.mean(dim=1)
    return q_hat
```

## Testing Checklist

1. **Backward compatibility:** With `sdr_beta: 0`, behavior must be identical to current SDR-ASBS. Run a short training on MoG25 and verify the loss curve matches.

2. **Weight sanity:** With `sdr_beta: 0.5` on MoG25 (which has 25 equal modes), log the SDR-DARW weights. Early in training when mode concentration hasn't set in, weights should be close to uniform. As concentration develops, weights for minority-mode particles should increase.

3. **Gradient check:** Run one step, verify that gradients flow correctly through the weighted loss. The weights are detached (computed in `populate_buffer` with `@torch.no_grad()`), so they should not appear in the computational graph.

4. **Full run:** Train on MoG25 and DW4 with `sdr_beta ∈ {0, 0.3, 0.5, 0.7, 1.0}` and compare mode coverage metrics.

## File Changes Summary

| File | Change |
|------|--------|
| `adjoint_samplers/components/sdr_matcher.py` | Add `sdr_beta`, `darw_weight_clip` params; add `_compute_darw_weights` method; modify `populate_buffer` to store weights |
| `adjoint_samplers/components/matcher.py` | No changes to base classes needed if you override `prepare_target` in the SDR matcher |
| `adjoint_samplers/train_loop.py` | Handle 3-value return from `prepare_target`; apply weights in loss |
| `configs/matcher/ksd_darw_adjoint_ve.yaml` | New config file |
| Various experiment configs | Add `sdr_beta` parameter |
| `train.py` | Add SDR-DARW weight logging |
