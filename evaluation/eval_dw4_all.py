"""
eval_dw4_all.py — Unified DW4 evaluation with corrected metrics.

Methods: ASBS, KSD-ASBS (λ=1.0), iDEM, pDEM
Metrics: energy_W2, Rg_W2, dist_W2 (fixed √), KSD²
5 eval seeds × 2000 samples each.

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python -u evaluation/eval_dw4_all.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import datetime
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
N_PARTICLES = 4
SPATIAL_DIM = 2
DIM = N_PARTICLES * SPATIAL_DIM  # 8

METRIC_KEYS = ['energy_w2', 'rg_w2', 'dist_w2', 'ksd_squared', 'mean_energy']


# ======================================================================
# Metrics
# ======================================================================

def radius_of_gyration(x, n_particles, spatial_dim):
    B = x.shape[0]
    coords = x.view(B, n_particles, spatial_dim)
    com = coords.mean(dim=1, keepdim=True)
    Rg = ((coords - com) ** 2).sum(dim=(1, 2)) / n_particles
    return Rg.sqrt()


@torch.no_grad()
def compute_metrics(samples, energy, ref_samples):
    metrics = {}
    device = samples.device

    valid_mask = ~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))
    valid_samples = samples[valid_mask]
    metrics['valid_count'] = int(valid_samples.shape[0])
    metrics['nan_count'] = int((~valid_mask).sum().item())

    if valid_samples.shape[0] < 10:
        print(f"  WARNING: Only {valid_samples.shape[0]} valid samples!")
        return metrics

    ref = ref_samples.to(device)
    B = min(valid_samples.shape[0], len(ref))
    idx_g = torch.randperm(valid_samples.shape[0], device=device)[:B]
    idx_r = torch.randperm(len(ref), device=device)[:B]
    gen = valid_samples[idx_g]
    ref_sub = ref[idx_r]

    # Energy W2
    gen_E = energy.eval(gen)
    ref_E = energy.eval(ref_sub)
    e_valid = ~(torch.isnan(gen_E) | torch.isinf(gen_E) | torch.isnan(ref_E) | torch.isinf(ref_E))
    try:
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E[e_valid].cpu().numpy(), gen_E[e_valid].cpu().numpy()) ** 0.5
        )
    except Exception as e:
        print(f"  energy_w2 failed: {e}")
        metrics['energy_w2'] = float('nan')

    metrics['mean_energy'] = float(gen_E[~torch.isnan(gen_E)].mean().item())

    # dist_W2 (fixed √)
    try:
        gen_dist = interatomic_dist(gen, N_PARTICLES, SPATIAL_DIM)
        ref_dist = interatomic_dist(ref_sub, N_PARTICLES, SPATIAL_DIM)
        metrics['dist_w2'] = float(pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        ) ** 0.5)
    except Exception as e:
        print(f"  dist_w2 failed: {e}")
        metrics['dist_w2'] = float('nan')

    # Rg_W2
    try:
        gen_rg = radius_of_gyration(gen, N_PARTICLES, SPATIAL_DIM)
        ref_rg = radius_of_gyration(ref_sub, N_PARTICLES, SPATIAL_DIM)
        metrics['rg_w2'] = float(
            pot.emd2_1d(ref_rg.cpu().numpy(), gen_rg.cpu().numpy()) ** 0.5
        )
    except Exception as e:
        print(f"  rg_w2 failed: {e}")
        metrics['rg_w2'] = float('nan')

    # KSD²
    try:
        with torch.enable_grad():
            s_req = gen[:min(2000, len(gen))].detach().requires_grad_(True)
            E_req = energy.eval(s_req)
            scores = -torch.autograd.grad(E_req.sum(), s_req)[0]
            s_req = s_req.detach()
        ell = median_bandwidth(s_req)
        metrics['ksd_squared'] = float(compute_ksd_squared(s_req, scores.detach(), ell).item())
    except Exception as e:
        print(f"  KSD failed: {e}")
        metrics['ksd_squared'] = float('nan')

    return metrics


# ======================================================================
# ASBS / KSD-ASBS model loader
# ======================================================================

def load_asbs_model(exp_dir, ckpt_name, device):
    exp_dir = Path(exp_dir)
    cfg = OmegaConf.load(exp_dir / 'config.yaml')
    cfg_r = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    energy = hydra.utils.instantiate(cfg_r.energy, device=device)
    source = hydra.utils.instantiate(cfg_r.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg_r.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg_r.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(exp_dir / 'checkpoints' / ckpt_name, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    ts_cfg = {
        't0': float(cfg_r.timesteps.t0),
        't1': float(cfg_r.timesteps.t1),
        'steps': int(cfg_r.nfe),
        'rescale_t': cfg_r.get('rescale_t', None),
    }
    return sde, source, energy, ts_cfg


@torch.no_grad()
def generate_asbs_samples(sde, source, ts_cfg, n_samples, device, batch_size=500):
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1)
        n += b
    return torch.cat(x1_list)[:n_samples]


# ======================================================================
# DEM model loader
# ======================================================================

def load_dem_model(exp_name, device):
    sys.path.insert(0, str(PROJECT_ROOT / 'baseline_models' / 'dem'))
    from dem.models.dem_module import DEMLitModule
    from dem.energies.base_prior import MeanFreePrior

    exp_dir = PROJECT_ROOT / 'results' / exp_name
    cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')
    energy_function = hydra.utils.instantiate(cfg.energy)
    model = hydra.utils.instantiate(cfg.model, energy_function=energy_function)

    ckpt_path = exp_dir / 'checkpoints' / 'last.ckpt'
    print(f"  {exp_name} checkpoint: {ckpt_path.name}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    noise_schedule = model.noise_schedule
    scale = noise_schedule.h(torch.tensor(1.0)).item() ** 0.5
    model.prior = MeanFreePrior(
        n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
        device=device, scale=scale,
    )
    return model


@torch.no_grad()
def generate_dem_samples(model, n_samples, device):
    return model.generate_samples(
        num_samples=n_samples,
        diffusion_scale=model.diffusion_scale,
    ).to(device)


# ======================================================================
# Eval driver
# ======================================================================

def eval_seeds(method_name, sample_fn, energy, ref_samples):
    print(f"\n{'='*70}", flush=True)
    print(f"  {method_name}", flush=True)
    print(f"{'='*70}", flush=True)

    seed_results = []
    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed * 7777)
        np.random.seed(seed * 7777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed * 7777)

        print(f"\n  Seed {seed}:", flush=True)
        samples = sample_fn()
        samples = remove_mean(samples, N_PARTICLES, SPATIAL_DIM)
        m = compute_metrics(samples, energy, ref_samples)
        dt = time.time() - t0

        seed_results.append(m)
        parts = []
        for k in METRIC_KEYS:
            if k in m:
                if k == 'ksd_squared':
                    parts.append(f"KSD²={m[k]:.4f}")
                elif k == 'mean_energy':
                    parts.append(f"<E>={m[k]:.2f}")
                else:
                    parts.append(f"{k}={m[k]:.4f}")
        print(f"    {' | '.join(parts)}  [{dt:.1f}s]", flush=True)

    print(f"\n  --- {method_name} mean±std ---", flush=True)
    for key in METRIC_KEYS:
        vals = [m[key] for m in seed_results if key in m and np.isfinite(m[key])]
        if vals:
            print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    return seed_results


def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 70, flush=True)
    print("DW4 Unified Evaluation — ASBS, KSD-ASBS (λ=1), AS, iDEM, pDEM", flush=True)
    print("Metrics: energy_W2, Rg_W2, dist_W2 (fixed √), KSD²", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    # Reference
    ref_path = PROJECT_ROOT / 'data' / 'test_split_DW4.npy'
    ref_np = np.load(ref_path, allow_pickle=True)
    ref_samples = remove_mean(
        torch.tensor(ref_np, dtype=torch.float32), N_PARTICLES, SPATIAL_DIM
    )
    print(f"Reference: {ref_samples.shape}", flush=True)

    # Energy (shared)
    energy_cfg = OmegaConf.create({
        '_target_': 'adjoint_samplers.energies.double_well_energy.DoubleWellEnergy',
        'dim': DIM, 'n_particles': N_PARTICLES,
    })
    energy = hydra.utils.instantiate(energy_cfg, device=DEVICE)

    all_results = {}

    # ---- 1. ASBS ----
    asbs_dir = PROJECT_ROOT / 'results' / 'dw4_asbs' / 'seed_0'
    print(f"ASBS dir: {asbs_dir}", flush=True)
    sde, source, _, ts_cfg = load_asbs_model(asbs_dir, 'checkpoint_latest.pt', DEVICE)
    all_results['ASBS'] = eval_seeds(
        'ASBS',
        lambda: generate_asbs_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE),
        energy, ref_samples,
    )
    del sde, source
    torch.cuda.empty_cache()

    # ---- 2. KSD-ASBS (λ=1.0) ----
    ksd_dir = PROJECT_ROOT / 'results' / 'dw4_ksd_asbs' / 'seed_0'
    print(f"KSD-ASBS dir: {ksd_dir}", flush=True)
    sde, source, _, ts_cfg = load_asbs_model(ksd_dir, 'checkpoint_latest.pt', DEVICE)
    all_results['KSD-ASBS'] = eval_seeds(
        'KSD-ASBS (λ=1.0)',
        lambda: generate_asbs_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE),
        energy, ref_samples,
    )
    del sde, source
    torch.cuda.empty_cache()

    # ---- 3. AS (Adjoint Sampler, no adjoint matching) ----
    as_dir = PROJECT_ROOT / 'results' / 'dw4_as' / 'seed_0'
    print(f"AS dir: {as_dir}", flush=True)
    sde, source, _, ts_cfg = load_asbs_model(as_dir, 'checkpoint_latest.pt', DEVICE)
    all_results['AS'] = eval_seeds(
        'AS',
        lambda: generate_asbs_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE),
        energy, ref_samples,
    )
    del sde, source
    torch.cuda.empty_cache()

    # ---- 4. iDEM ----
    idem_model = load_dem_model('dw4_idem', DEVICE)
    all_results['iDEM'] = eval_seeds(
        'iDEM',
        lambda: generate_dem_samples(idem_model, N_SAMPLES, DEVICE),
        energy, ref_samples,
    )
    del idem_model
    torch.cuda.empty_cache()

    # ---- 5. pDEM ----
    pdem_model = load_dem_model('dw4_pdem', DEVICE)
    all_results['pDEM'] = eval_seeds(
        'pDEM',
        lambda: generate_dem_samples(pdem_model, N_SAMPLES, DEVICE),
        energy, ref_samples,
    )
    del pdem_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n\n{'='*80}", flush=True)
    print("SUMMARY — DW4 Unified Evaluation (fixed metrics)", flush=True)
    print(f"{'='*80}\n", flush=True)

    header = (f"{'Method':<15} {'Seed':<5} {'energy_W2':>10} {'Rg_W2':>10} "
              f"{'dist_W2':>10} {'KSD²':>12} {'<E>':>10}")
    print(header, flush=True)
    print("─" * len(header), flush=True)

    for method_name, seed_results in all_results.items():
        for i, r in enumerate(seed_results):
            def g(k, fmt='.4f'):
                v = r.get(k, float('nan'))
                if np.isfinite(v):
                    return f"{v:{fmt}}"
                return "NaN"
            print(
                f"{method_name:<15} {i:<5} {g('energy_w2'):>10} {g('rg_w2'):>10} "
                f"{g('dist_w2'):>10} {g('ksd_squared', '.4f'):>12} {g('mean_energy', '.2f'):>10}",
                flush=True,
            )
        for key in METRIC_KEYS:
            vals = [m[key] for m in seed_results if key in m and np.isfinite(m[key])]
            if vals:
                print(f"{'':>15} {'μ±σ':<5} {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)
        print(flush=True)

    # Save
    out_path = PROJECT_ROOT / 'evaluation' / 'eval_dw4_all_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
