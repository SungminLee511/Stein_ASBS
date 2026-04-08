"""
eval_dw4_baselines.py — Evaluate DW4 baselines (AS, iDEM, pDEM, DGFS)
on the same metrics as ASBS / KSD-ASBS: energy_W2, eq_W2, dist_W2, KSD².

ESS via Girsanov is only computed for AS (same SDE framework as ASBS).
For iDEM/pDEM/DGFS, ESS is not applicable (different generative process).

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python -u evaluation/eval_dw4_baselines.py
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
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils

# Import ESS utilities (for AS only)
from evaluation.evaluate_ess import sdeint_with_noise, compute_girsanov_log_weights, compute_ess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
ESS_BATCH_SIZE = 500
N_PARTICLES = 4
SPATIAL_DIM = 2


# ======================================================================
# Shared metric computation (same as eval_dw4_full.py)
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
# 1. AS (Adjoint Sampler) — same framework as ASBS
# ======================================================================

def load_as_model(device):
    """Load AS model and return (sde, source, energy, ts_cfg)."""
    exp_dir = PROJECT_ROOT / 'results' / 'dw4_as' / 'seed_0'
    cfg = OmegaConf.load(exp_dir / 'config.yaml')
    cfg_r = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    energy = hydra.utils.instantiate(cfg_r.energy, device=device)
    source = hydra.utils.instantiate(cfg_r.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg_r.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg_r.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
    print(f"  AS checkpoint: {ckpt_path.name}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
def generate_as_samples(sde, source, ts_cfg, n_samples, device, batch_size=500):
    """Generate terminal samples from AS/ASBS-style SDE."""
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


@torch.no_grad()
def compute_ess_for_seed(sde, source, ts_cfg, n_samples, device, batch_size=500):
    """Compute ESS via Girsanov importance weights."""
    all_log_w = []
    n = 0
    sde.eval()
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        ts = train_utils.get_timesteps(**ts_cfg).to(device)
        states, noises, dts = sdeint_with_noise(sde, x0, ts)
        log_w = compute_girsanov_log_weights(sde, states, noises, dts, ts)
        all_log_w.append(log_w)
        n += b
        del states, noises, dts
        torch.cuda.empty_cache()
    log_weights = torch.cat(all_log_w)[:n_samples]
    ess = compute_ess(log_weights)
    return ess, ess / n_samples


# ======================================================================
# 2. DEM (iDEM / pDEM) — Lightning framework
# ======================================================================

def load_dem_model(exp_name, device):
    """Load DEM (iDEM or pDEM) model. Returns (model, energy_adj) where
    energy_adj is the adjoint_samplers energy for KSD/W2 computation."""
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
# 3. DGFS (GFlowNet)
# ======================================================================

def load_dgfs_model(device):
    """Load DGFS model for DW4."""
    sys.path.insert(0, str(PROJECT_ROOT / 'baseline_models' / 'dgfs'))

    from gflownet.gflownet import get_alg
    from target.distribution.dw4 import DW4

    exp_dir = PROJECT_ROOT / 'results' / 'dw4_dgfs'
    cfg = OmegaConf.load(exp_dir / '.hydra' / 'config.yaml')

    # Add missing keys
    cfg.data_ndim = 8
    cfg.t_end = cfg.dt * cfg.N
    if 'weight_decay' not in cfg:
        cfg.weight_decay = cfg.get('wd', 1e-7)
    if 'subtb_lambda' not in cfg:
        cfg.subtb_lambda = cfg.get('stlam', 2.0)
    if 'batch_size' not in cfg:
        cfg.batch_size = cfg.get('bs', 256)

    data = DW4()
    model = get_alg(cfg, task=data).to(device)

    ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
    print(f"  DGFS checkpoint: {ckpt_path.name}", flush=True)
    model.load(str(ckpt_path))
    model.eval()

    return model, data, cfg


@torch.no_grad()
def generate_dgfs_samples(model, data, cfg, n_samples, device):
    """Generate samples from DGFS model."""
    from gflownet.gflownet import sample_traj

    logreward_fn = lambda x: -data.energy(x)
    traj, info = sample_traj(model, cfg, logreward_fn, batch_size=n_samples)
    samples = traj[-1][1].to(device)
    return samples


# ======================================================================
# Main
# ======================================================================

def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 70, flush=True)
    print("DW4 Baseline Evaluation — energy_W2, eq_W2, dist_W2, KSD²", flush=True)
    print(f"Methods: AS, iDEM, pDEM, DGFS", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    # Load reference samples
    ref_path = PROJECT_ROOT / 'data' / 'test_split_DW4.npy'
    ref_np = np.load(ref_path, allow_pickle=True)
    ref_samples = remove_mean(torch.tensor(ref_np, dtype=torch.float32), N_PARTICLES, SPATIAL_DIM)
    print(f"Reference: {ref_samples.shape}", flush=True)

    # Load adjoint_samplers energy for metrics (shared across all methods)
    energy_cfg = OmegaConf.create({
        '_target_': 'adjoint_samplers.energies.double_well_energy.DoubleWellEnergy',
        'dim': 8,
        'n_particles': 4,
    })
    energy = hydra.utils.instantiate(energy_cfg, device=DEVICE)

    all_results = {}

    # ==================================================================
    # 1. AS (Adjoint Sampler) — with ESS
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  AS (Adjoint Sampler)", flush=True)
    print(f"{'='*70}", flush=True)

    sde_as, source_as, _, ts_cfg_as = load_as_model(DEVICE)
    seed_results = []

    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n  Seed {seed}:", flush=True)

        samples = generate_as_samples(sde_as, source_as, ts_cfg_as, N_SAMPLES, DEVICE)
        samples = remove_mean(samples, N_PARTICLES, SPATIAL_DIM)

        print(f"    Computing W2 metrics...", flush=True)
        energy_w2, eq_w2, dist_w2 = compute_w2_metrics(samples, energy, ref_samples)

        print(f"    Computing KSD²...", flush=True)
        ksd2 = compute_ksd(samples, energy)

        print(f"    Computing ESS ({N_SAMPLES} trajectories)...", flush=True)
        torch.manual_seed(seed + 10000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 10000)
        ess, ess_frac = compute_ess_for_seed(
            sde_as, source_as, ts_cfg_as, N_SAMPLES, DEVICE, ESS_BATCH_SIZE
        )

        dt = time.time() - t0
        result = {
            'seed': seed,
            'energy_w2': energy_w2,
            'eq_w2': eq_w2,
            'dist_w2': dist_w2,
            'ksd_squared': ksd2,
            'ess': ess,
            'ess_fraction': ess_frac,
        }
        seed_results.append(result)
        print(f"    energy_W2={energy_w2:.4f}  eq_W2={eq_w2:.4f}  dist_W2={dist_w2:.6f}  "
              f"KSD²={ksd2:.6f}  ESS={ess:.2f} ({ess_frac*100:.4f}%)  [{dt:.1f}s]", flush=True)

    all_results['AS'] = seed_results
    for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'ess']:
        vals = [r[key] for r in seed_results]
        print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    del sde_as, source_as
    torch.cuda.empty_cache()

    # ==================================================================
    # 2. iDEM — no ESS
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  iDEM", flush=True)
    print(f"{'='*70}", flush=True)

    dem_idem = load_dem_model('dw4_idem', DEVICE)
    seed_results = []

    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n  Seed {seed}:", flush=True)

        samples = generate_dem_samples(dem_idem, N_SAMPLES, DEVICE)
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
            'ess': None,
            'ess_fraction': None,
        }
        seed_results.append(result)
        print(f"    energy_W2={energy_w2:.4f}  eq_W2={eq_w2:.4f}  dist_W2={dist_w2:.6f}  "
              f"KSD²={ksd2:.6f}  [{dt:.1f}s]", flush=True)

    all_results['iDEM'] = seed_results
    for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
        vals = [r[key] for r in seed_results]
        print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    del dem_idem
    torch.cuda.empty_cache()

    # ==================================================================
    # 3. pDEM — no ESS
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  pDEM", flush=True)
    print(f"{'='*70}", flush=True)

    dem_pdem = load_dem_model('dw4_pdem', DEVICE)
    seed_results = []

    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n  Seed {seed}:", flush=True)

        samples = generate_dem_samples(dem_pdem, N_SAMPLES, DEVICE)
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
            'ess': None,
            'ess_fraction': None,
        }
        seed_results.append(result)
        print(f"    energy_W2={energy_w2:.4f}  eq_W2={eq_w2:.4f}  dist_W2={dist_w2:.6f}  "
              f"KSD²={ksd2:.6f}  [{dt:.1f}s]", flush=True)

    all_results['pDEM'] = seed_results
    for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
        vals = [r[key] for r in seed_results]
        print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    del dem_pdem
    torch.cuda.empty_cache()

    # ==================================================================
    # 4. DGFS — no ESS
    # ==================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  DGFS", flush=True)
    print(f"{'='*70}", flush=True)

    dgfs_model, dgfs_data, dgfs_cfg = load_dgfs_model(DEVICE)
    seed_results = []

    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n  Seed {seed}:", flush=True)

        samples = generate_dgfs_samples(dgfs_model, dgfs_data, dgfs_cfg, N_SAMPLES, DEVICE)
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
            'ess': None,
            'ess_fraction': None,
        }
        seed_results.append(result)
        print(f"    energy_W2={energy_w2:.4f}  eq_W2={eq_w2:.4f}  dist_W2={dist_w2:.6f}  "
              f"KSD²={ksd2:.6f}  [{dt:.1f}s]", flush=True)

    all_results['DGFS'] = seed_results
    for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
        vals = [r[key] for r in seed_results]
        print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    del dgfs_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n\n{'='*70}", flush=True)
    print("SUMMARY — DW4 Baseline Evaluation", flush=True)
    print(f"{'='*70}", flush=True)

    header = (f"{'Method':<12} {'Seed':<6} {'energy_W2':>10} {'eq_W2':>10} "
              f"{'dist_W2':>10} {'KSD²':>10} {'ESS':>10} {'ESS%':>8}")
    print(f"\n{header}", flush=True)
    print("─" * len(header), flush=True)

    for method_name, seed_results in all_results.items():
        for r in seed_results:
            ess_str = f"{r['ess']:>10.2f}" if r['ess'] is not None else f"{'N/A':>10}"
            ess_pct = f"{r['ess_fraction']*100:>7.4f}%" if r['ess_fraction'] is not None else f"{'N/A':>8}"
            print(
                f"{method_name:<12} {r['seed']:<6} "
                f"{r['energy_w2']:>10.4f} {r['eq_w2']:>10.4f} {r['dist_w2']:>10.6f} "
                f"{r['ksd_squared']:>10.6f} {ess_str} {ess_pct}",
                flush=True,
            )
        # Mean ± std
        for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared']:
            vals = [r[key] for r in seed_results]
            print(f"{'':>12} {'mean±std':<6} {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)
        if seed_results[0]['ess'] is not None:
            ess_vals = [r['ess'] for r in seed_results]
            print(f"{'':>12} {'mean±std':<6} ess: {np.mean(ess_vals):.2f} ± {np.std(ess_vals):.2f}", flush=True)
        print(flush=True)

    # Save
    out_path = PROJECT_ROOT / 'evaluation' / 'eval_dw4_baselines_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
