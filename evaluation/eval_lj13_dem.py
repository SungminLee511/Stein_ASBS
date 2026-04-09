"""
eval_lj13_dem.py — Evaluate LJ13 iDEM and pDEM baselines
on: energy_W2, eq_W2, dist_W2, KSD².

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python -u evaluation/eval_lj13_dem.py
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

from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
N_PARTICLES = 13
SPATIAL_DIM = 3


# ======================================================================
# Metric computation (adapted from eval_dw4_baselines.py)
# ======================================================================

@torch.no_grad()
def compute_w2_metrics(samples, energy, ref_samples):
    """Compute energy_W2, eq_W2, dist_W2."""
    B = samples.shape[0]
    device = samples.device

    idxs = torch.randperm(len(ref_samples))[:B]
    ref = ref_samples[idxs].to(device)

    # energy_W2
    gen_E = energy.eval(samples)
    ref_E = energy.eval(ref)
    energy_w2 = float(pot.emd2_1d(ref_E.cpu().numpy(), gen_E.cpu().numpy()) ** 0.5)

    # dist_W2
    gen_dist = interatomic_dist(samples, N_PARTICLES, SPATIAL_DIM)
    ref_dist = interatomic_dist(ref, N_PARTICLES, SPATIAL_DIM)
    dist_w2 = float(pot.emd2_1d(
        gen_dist.cpu().numpy().reshape(-1),
        ref_dist.cpu().numpy().reshape(-1),
    ))

    # eq_W2
    M = dist_point_clouds(
        samples.reshape(-1, N_PARTICLES, SPATIAL_DIM).cpu(),
        ref.reshape(-1, N_PARTICLES, SPATIAL_DIM).cpu(),
    )
    a = torch.ones(M.shape[0]) / M.shape[0]
    b = torch.ones(M.shape[0]) / M.shape[0]
    eq_w2 = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)

    return energy_w2, eq_w2, dist_w2


def compute_ksd(samples, energy):
    """Compute KSD² with score from energy gradient."""
    with torch.enable_grad():
        s = samples.detach().requires_grad_(True)
        E = energy.eval(s)
        scores = -torch.autograd.grad(E.sum(), s)[0]
        s = s.detach()
    ell = median_bandwidth(s)
    ksd2 = compute_ksd_squared(s, scores.detach(), ell).item()
    return ksd2


# ======================================================================
# DEM (iDEM / pDEM) — Lightning framework
# ======================================================================

def load_dem_model(exp_name, device):
    """Load DEM (iDEM or pDEM) model."""
    sys.path.insert(0, str(PROJECT_ROOT / 'baseline_models' / 'dem'))

    from dem.models.dem_module import DEMLitModule
    from dem.energies.base_prior import MeanFreePrior

    exp_dir = PROJECT_ROOT / 'results' / exp_name
    cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')

    # Instantiate DEM energy function
    energy_function = hydra.utils.instantiate(cfg.energy)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, energy_function=energy_function)

    # Load checkpoint
    ckpt_path = exp_dir / 'checkpoints' / 'last.ckpt'
    print(f"  {exp_name} checkpoint: {ckpt_path.name}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    # Manually set up prior (normally done in Lightning setup())
    noise_schedule = model.noise_schedule
    scale = noise_schedule.h(torch.tensor(1.0)).item() ** 0.5
    model.prior = MeanFreePrior(
        n_particles=N_PARTICLES, spatial_dim=SPATIAL_DIM,
        device=device, scale=scale,
    )

    return model


@torch.no_grad()
def generate_dem_samples(model, n_samples, device):
    """Generate samples from DEM model."""
    samples = model.generate_samples(
        num_samples=n_samples,
        diffusion_scale=model.diffusion_scale,
    )
    return samples.to(device)


# ======================================================================
# Main
# ======================================================================

def eval_method(method_name, exp_name, energy, ref_samples):
    """Evaluate a single DEM method."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {method_name}", flush=True)
    print(f"{'='*70}", flush=True)

    model = load_dem_model(exp_name, DEVICE)
    seed_results = []

    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n  Seed {seed}:", flush=True)

        samples = generate_dem_samples(model, N_SAMPLES, DEVICE)
        samples = remove_mean(samples, N_PARTICLES, SPATIAL_DIM)

        print(f"    Computing W2 metrics...", flush=True)
        energy_w2, eq_w2, dist_w2 = compute_w2_metrics(samples, energy, ref_samples)

        print(f"    Computing KSD²...", flush=True)
        ksd2 = compute_ksd(samples, energy)

        dt = time.time() - t0
        result = {
            'seed': seed,
            'energy_w2': energy_w2,
            'eq_w2': eq_w2,
            'dist_w2': dist_w2,
            'ksd_squared': ksd2,
        }
        seed_results.append(result)
        print(f"    energy_W2={energy_w2:.4f}  eq_W2={eq_w2:.4f}  dist_W2={dist_w2:.6f}  "
              f"KSD²={ksd2:.6f}  [{dt:.1f}s]", flush=True)

    for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
        vals = [r[key] for r in seed_results]
        print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    del model
    torch.cuda.empty_cache()

    return seed_results


def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 70, flush=True)
    print("LJ13 DEM Baseline Evaluation — energy_W2, eq_W2, dist_W2, KSD²", flush=True)
    print(f"Methods: iDEM, pDEM", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    # Load reference samples
    ref_path = PROJECT_ROOT / 'data' / 'test_split_LJ13-1000.npy'
    ref_np = np.load(ref_path, allow_pickle=True)
    ref_samples = remove_mean(torch.tensor(ref_np, dtype=torch.float32), N_PARTICLES, SPATIAL_DIM)
    print(f"Reference: {ref_samples.shape}", flush=True)

    # Load adjoint_samplers energy for metrics
    energy_cfg = OmegaConf.create({
        '_target_': 'adjoint_samplers.energies.lennard_jones_energy.LennardJonesEnergy',
        'dim': 39,
        'n_particles': 13,
    })
    energy = hydra.utils.instantiate(energy_cfg, device=DEVICE)

    all_results = {}

    # 1. iDEM
    all_results['iDEM'] = eval_method('iDEM', 'lj13_idem', energy, ref_samples)

    # 2. pDEM
    all_results['pDEM'] = eval_method('pDEM', 'lj13_pdem', energy, ref_samples)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n\n{'='*70}", flush=True)
    print("SUMMARY — LJ13 DEM Baseline Evaluation", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"{'Method':<12} {'Seed':<6} {'energy_W2':>10} {'eq_W2':>10} "
              f"{'dist_W2':>10} {'KSD²':>10}")
    print(f"\n{header}", flush=True)
    print("─" * len(header), flush=True)

    for method_name, seed_results in all_results.items():
        for r in seed_results:
            print(
                f"{method_name:<12} {r['seed']:<6} "
                f"{r['energy_w2']:>10.4f} {r['eq_w2']:>10.4f} {r['dist_w2']:>10.6f} "
                f"{r['ksd_squared']:>10.6f}",
                flush=True,
            )
        for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
            vals = [r[key] for r in seed_results]
            print(f"{'':>12} {'mean±std':<6} {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)
        print(flush=True)

    # Save
    out_path = PROJECT_ROOT / 'evaluation' / 'eval_lj13_dem_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
