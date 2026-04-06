"""
ESS Evaluation via Girsanov Importance Weights.

Computes Effective Sample Size (ESS) for all trained experiments.
ESS measures how well the controlled SDE's path measure covers the reference —
higher ESS means more uniform importance weights and better mode coverage.

Usage:
    python -u evaluation/evaluate_ess.py > evaluation/ess_eval_log.txt 2>&1
"""

import sys
import os
import json
import time
import datetime
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adjoint_samplers.components.sde import ControlledSDE, BaseSDE
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Core ESS functions
# ====================================================================

@torch.no_grad()
def sdeint_with_noise(
    sde: BaseSDE,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple:
    """Forward SDE integration that also returns noise realizations.

    Mirrors the original sdeint exactly, but saves the noise eps_k at each step.

    Returns:
        states:  list of (B, D) tensors — trajectory [x_0, x_1, ..., x_K]
        noises:  list of (B, D) tensors — noise realizations [ε_0, ..., ε_{K-1}]
        dts:     list of scalar tensors — time steps [Δt_0, ..., Δt_{K-1}]
    """
    T = len(timesteps)
    assert T > 1

    sde.train(False)
    state = state0.clone()
    states = [state0]
    noises = []
    dts = []

    for i in range(T - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        # Generate noise BEFORE computing diffusion — same as sdeint
        eps = sde.randn_like(state)  # ε_k ~ N(0, I) — uses COM-free noise for Graph SDEs

        drift = sde.drift(t, state) * dt
        diffusion = sde.diff(t) * dt.sqrt() * eps

        d_state = drift + diffusion
        state = sde.propagate(state, d_state)

        states.append(state)
        noises.append(eps)
        dts.append(dt)

    return states, noises, dts


@torch.no_grad()
def compute_girsanov_log_weights(
    sde: ControlledSDE,
    states: list,
    noises: list,
    dts: list,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Compute log-importance-weight for each trajectory via discrete Girsanov formula.

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

        # Controller output: u_θ(t_k, x_k)
        u_k = sde.u(t_k, x_k)  # (B, D)

        # Diffusion coefficient from reference SDE
        g_k = sde.ref_sde.diff(t_k)  # scalar or (B,) or (B,1)

        # Ensure g_k is broadcastable with u_k
        if g_k.dim() == 0:
            g_k = g_k.unsqueeze(0)
        while g_k.dim() < u_k.dim():
            g_k = g_k.unsqueeze(-1)

        # Stochastic integral term: g_k * sqrt(dt) * u^T ε
        stoch_term = (g_k * dt_k.sqrt() * u_k * eps_k).sum(dim=-1)  # (B,)

        # Control cost term: ½ g_k² ||u||² dt
        cost_term = 0.5 * (g_k ** 2 * (u_k ** 2).sum(dim=-1, keepdim=True) * dt_k).squeeze(-1)  # (B,)

        log_w = log_w - stoch_term - cost_term

    return log_w


def compute_ess(log_weights: torch.Tensor) -> float:
    """Compute Effective Sample Size from log-importance-weights.

    ESS = 1 / Σ w_i²  where w_i are normalized (sum to 1).
    Uses logsumexp for numerical stability.

    Args:
        log_weights: (N,) tensor of log W_i

    Returns:
        ess: scalar, in range [1, N]
    """
    # Normalize: log w_i - log(Σ exp(log w_j))
    log_w_normalized = log_weights - torch.logsumexp(log_weights, dim=0)

    # ESS = 1 / Σ w_i²  = exp(-log(Σ exp(2 log w_i)))
    log_sum_sq = torch.logsumexp(2 * log_w_normalized, dim=0)
    ess = torch.exp(-log_sum_sq).item()

    return ess


@torch.no_grad()
def evaluate_ess(
    sde: ControlledSDE,
    source,
    timesteps_cfg: dict,
    n_samples: int = 10000,
    batch_size: int = 500,
    device: str = 'cuda',
) -> dict:
    """Evaluate ESS for a trained model.

    Args:
        sde: trained ControlledSDE
        source: source distribution
        timesteps_cfg: dict for train_utils.get_timesteps
        n_samples: total trajectories to evaluate
        batch_size: trajectories per batch
        device: torch device

    Returns:
        dict with ESS metrics
    """
    all_log_weights = []
    all_control_costs = []

    sde.eval()
    n_generated = 0
    batch_idx = 0

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
            while g_k.dim() < u_k.dim():
                g_k = g_k.unsqueeze(-1)
            control_cost += 0.5 * (g_k ** 2 * (u_k ** 2).sum(dim=-1, keepdim=True) * dt_k).squeeze(-1)
        all_control_costs.append(control_cost)

        n_generated += b
        batch_idx += 1

        # Free intermediate states to save memory
        del states, noises, dts
        torch.cuda.empty_cache()

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


# ====================================================================
# Experiment loading (adapted from evaluate_all.py)
# ====================================================================

def load_experiment(exp_dir, device, ckpt_name='checkpoint_latest.pt'):
    """Load a trained experiment from its result directory.

    Args:
        exp_dir: path to experiment directory (contains config.yaml + checkpoints/)
        device: torch device
        ckpt_name: checkpoint filename to load

    Returns:
        dict with 'sde', 'source', 'ts_cfg', 'cfg' or None on failure
    """
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / 'config.yaml'
    ckpt_path = exp_dir / 'checkpoints' / ckpt_name

    if not cfg_path.exists():
        cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        print(f"    WARNING: No config found at {exp_dir}", flush=True)
        return None
    if not ckpt_path.exists():
        print(f"    WARNING: No checkpoint found at {ckpt_path}", flush=True)
        return None

    cfg = OmegaConf.load(cfg_path)

    # Instantiate components
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    # Build timestep config
    if hasattr(cfg, 'timesteps'):
        ts_cfg = {
            't0': cfg.timesteps.t0, 't1': cfg.timesteps.t1,
            'steps': cfg.timesteps.steps, 'rescale_t': cfg.timesteps.rescale_t,
        }
    else:
        ts_cfg = {
            't0': 0.0, 't1': 1.0,
            'steps': cfg.get('nfe', 200),
            'rescale_t': cfg.get('rescale_t', None),
        }

    return {
        'sde': sde, 'source': source, 'ts_cfg': ts_cfg, 'cfg': cfg,
    }


# ====================================================================
# Experiment registry — all experiments from RESULTS.md (no ablations, no LJ38)
# ====================================================================

RESULTS_ROOT = PROJECT_ROOT / 'results'
BASELINES_ROOT = PROJECT_ROOT / 'baselines'

EXPERIMENTS = [
    # --- DW4 (8D) --- DONE, results in ess_eval_log_dw4.txt
    # {
    #     'name': 'DW4 — Baseline',
    #     'dir': RESULTS_ROOT / 'dw4_asbs' / 'seed_0',
    #     'ckpt': 'checkpoint_latest.pt',
    #     'group': 'DW4',
    # },
    # {
    #     'name': 'DW4 — KSD-ASBS (RBF, λ=1.0)',
    #     'dir': RESULTS_ROOT / 'dw4_ksd_asbs' / 'seed_0',
    #     'ckpt': 'checkpoint_latest.pt',
    #     'group': 'DW4',
    # },

    # --- Müller-Brown (2D) ---
    {
        'name': 'Müller-Brown — Baseline',
        'dir': RESULTS_ROOT / 'muller_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'Muller',
    },
    {
        'name': 'Müller-Brown — KSD-ASBS (RBF, λ=0.01)',
        'dir': RESULTS_ROOT / 'muller_ksd_asbs' / 'seed_1',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'Muller',
    },

    # --- RotGMM d=10 ---
    {
        'name': 'RotGMM d=10 — Baseline',
        'dir': RESULTS_ROOT / 'rotgmm10_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-10',
    },
    {
        'name': 'RotGMM d=10 — KSD-ASBS (RBF, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm10_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-10',
    },
    {
        'name': 'RotGMM d=10 — KSD-ASBS (IMQ, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm10_imq_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-10',
    },

    # --- RotGMM d=30 ---
    {
        'name': 'RotGMM d=30 — Baseline',
        'dir': RESULTS_ROOT / 'rotgmm30_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-30',
    },
    {
        'name': 'RotGMM d=30 — KSD-ASBS (RBF, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm30_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-30',
    },
    {
        'name': 'RotGMM d=30 — KSD-ASBS (IMQ, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm30_imq_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-30',
    },

    # --- RotGMM d=50 ---
    {
        'name': 'RotGMM d=50 — Baseline',
        'dir': RESULTS_ROOT / 'rotgmm50_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-50',
    },
    {
        'name': 'RotGMM d=50 — KSD-ASBS (RBF, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm50_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-50',
    },
    {
        'name': 'RotGMM d=50 — KSD-ASBS (IMQ, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm50_imq_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-50',
    },

    # --- RotGMM d=100 ---
    {
        'name': 'RotGMM d=100 — Baseline',
        'dir': RESULTS_ROOT / 'rotgmm100_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-100',
    },
    {
        'name': 'RotGMM d=100 — KSD-ASBS (RBF, λ=0.1)',
        'dir': RESULTS_ROOT / 'rotgmm100_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-100',
    },
    {
        'name': 'RotGMM d=100 — KSD-ASBS (IMQ, λ=1.0)',
        'dir': RESULTS_ROOT / 'rotgmm100_imq_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'RotGMM-100',
    },

    # --- BLogReg Australian (d=15) ---
    {
        'name': 'BLogReg Australian — Baseline',
        'dir': RESULTS_ROOT / 'blogreg_au_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'BLogReg-Au',
    },
    {
        'name': 'BLogReg Australian — KSD-ASBS',
        'dir': RESULTS_ROOT / 'blogreg_au_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'BLogReg-Au',
    },

    # --- BLogReg German (d=25) ---
    {
        'name': 'BLogReg German — Baseline',
        'dir': RESULTS_ROOT / 'blogreg_ge_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'BLogReg-Ge',
    },
    {
        'name': 'BLogReg German — KSD-ASBS',
        'dir': RESULTS_ROOT / 'blogreg_ge_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'BLogReg-Ge',
    },

    # --- LJ13 (39D) — EGNN, slow, placed last ---
    {
        'name': 'LJ13 — Baseline',
        'dir': RESULTS_ROOT / 'lj13_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
        'group': 'LJ13',
    },
    {
        'name': 'LJ13 — KSD-ASBS (RBF, λ=1.0, ckpt1550)',
        'dir': RESULTS_ROOT / 'lj13_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_1550.pt',
        'group': 'LJ13',
    },
]


# ====================================================================
# Main
# ====================================================================

def main():
    N_SAMPLES = 10000
    BATCH_SIZE = 500
    SEED = 0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 80, flush=True)
    print(f"ESS Evaluation — Girsanov Importance Weights", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, BATCH_SIZE={BATCH_SIZE}, SEED={SEED}, DEVICE={DEVICE}", flush=True)
    print(f"Total experiments: {len(EXPERIMENTS)}", flush=True)
    print("=" * 80, flush=True)

    all_results = {}
    total_start = time.time()

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        exp_dir = exp['dir']
        ckpt_name = exp['ckpt']
        group = exp['group']

        print(f"\n{'─' * 70}", flush=True)
        print(f"[{i+1}/{len(EXPERIMENTS)}] {exp_name}", flush=True)
        print(f"  Dir:  {exp_dir}", flush=True)
        print(f"  Ckpt: {ckpt_name}", flush=True)

        exp_start = time.time()

        try:
            # Load model
            loaded = load_experiment(exp_dir, DEVICE, ckpt_name)
            if loaded is None:
                print(f"  ❌ SKIPPED (load failed)", flush=True)
                all_results[exp_name] = {'status': 'skipped', 'group': group}
                continue

            print(f"  Model loaded. Running ESS evaluation ({N_SAMPLES} trajectories)...", flush=True)

            # Reset seed for reproducibility per experiment
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(SEED)

            # Run ESS evaluation
            ess_result = evaluate_ess(
                sde=loaded['sde'],
                source=loaded['source'],
                timesteps_cfg=loaded['ts_cfg'],
                n_samples=N_SAMPLES,
                batch_size=BATCH_SIZE,
                device=DEVICE,
            )

            exp_time = time.time() - exp_start

            # Store result
            result_entry = {
                'status': 'ok',
                'group': group,
                'time_sec': round(exp_time, 1),
                **ess_result,
            }
            all_results[exp_name] = result_entry

            # Live log
            print(f"  ✅ Done in {exp_time:.1f}s", flush=True)
            print(f"  ┌─────────────────────────────────────────────", flush=True)
            print(f"  │ ESS          = {ess_result['ess']:.2f}  ({ess_result['ess_fraction']*100:.4f}% of N={N_SAMPLES})", flush=True)
            print(f"  │ log W  mean  = {ess_result['log_weights_mean']:.4f}", flush=True)
            print(f"  │ log W  std   = {ess_result['log_weights_std']:.4f}", flush=True)
            print(f"  │ log W  range = [{ess_result['log_weights_min']:.4f}, {ess_result['log_weights_max']:.4f}]", flush=True)
            print(f"  │ ctrl cost    = {ess_result['control_cost_mean']:.4f} ± {ess_result['control_cost_std']:.4f}", flush=True)
            print(f"  └─────────────────────────────────────────────", flush=True)

            # Free GPU memory
            del loaded
            torch.cuda.empty_cache()

        except Exception as e:
            exp_time = time.time() - exp_start
            print(f"  ❌ ERROR after {exp_time:.1f}s: {e}", flush=True)
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'status': 'error', 'group': group, 'error': str(e)}
            torch.cuda.empty_cache()

    # ====================================================================
    # Summary table
    # ====================================================================
    total_time = time.time() - total_start
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"SUMMARY — ESS Results", flush=True)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Group results for comparison
    groups = {}
    for name, res in all_results.items():
        g = res.get('group', 'unknown')
        if g not in groups:
            groups[g] = []
        groups[g].append((name, res))

    header = f"{'Experiment':<50} {'ESS':>10} {'ESS/N%':>8} {'logW mean':>12} {'logW std':>10} {'ctrl cost':>12}"
    print(f"\n{header}", flush=True)
    print("─" * len(header), flush=True)

    for group_name in dict.fromkeys(r['group'] for r in all_results.values()):
        entries = groups.get(group_name, [])
        for name, res in entries:
            if res['status'] == 'ok':
                print(
                    f"  {name:<48} "
                    f"{res['ess']:>10.2f} "
                    f"{res['ess_fraction']*100:>7.4f}% "
                    f"{res['log_weights_mean']:>12.4f} "
                    f"{res['log_weights_std']:>10.4f} "
                    f"{res['control_cost_mean']:>12.4f}",
                    flush=True,
                )
            else:
                print(f"  {name:<48}  {'SKIPPED/ERROR':>10}", flush=True)
        print(flush=True)

    # ====================================================================
    # Save JSON
    # ====================================================================
    output_path = PROJECT_ROOT / 'evaluation' / 'ess_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
