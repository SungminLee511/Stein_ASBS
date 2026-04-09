"""
eval_lj13_as.py — Evaluate LJ13 AS (currently training) latest checkpoint.
Metrics: energy_W2, Rg_W2, dist_W2 (fixed √), KSD²
5 eval seeds × 2000 samples each.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, time, datetime, torch, numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import hydra, ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
N_PARTICLES = 13
SPATIAL_DIM = 3
DIM = 39

METRIC_KEYS = ['energy_w2', 'rg_w2', 'dist_w2', 'ksd_squared', 'mean_energy']


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

    gen_E = energy.eval(gen)
    ref_E = energy.eval(ref_sub)
    e_valid = ~(torch.isnan(gen_E) | torch.isinf(gen_E) | torch.isnan(ref_E) | torch.isinf(ref_E))
    try:
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E[e_valid].cpu().numpy(), gen_E[e_valid].cpu().numpy()) ** 0.5
        )
    except Exception as e:
        metrics['energy_w2'] = float('nan')

    metrics['mean_energy'] = float(gen_E[~torch.isnan(gen_E)].mean().item())

    try:
        gen_dist = interatomic_dist(gen, N_PARTICLES, SPATIAL_DIM)
        ref_dist = interatomic_dist(ref_sub, N_PARTICLES, SPATIAL_DIM)
        metrics['dist_w2'] = float(pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        ) ** 0.5)
    except Exception as e:
        metrics['dist_w2'] = float('nan')

    try:
        gen_rg = radius_of_gyration(gen, N_PARTICLES, SPATIAL_DIM)
        ref_rg = radius_of_gyration(ref_sub, N_PARTICLES, SPATIAL_DIM)
        metrics['rg_w2'] = float(
            pot.emd2_1d(ref_rg.cpu().numpy(), gen_rg.cpu().numpy()) ** 0.5
        )
    except Exception as e:
        metrics['rg_w2'] = float('nan')

    try:
        with torch.enable_grad():
            s_req = gen[:min(2000, len(gen))].detach().requires_grad_(True)
            E_req = energy.eval(s_req)
            scores = -torch.autograd.grad(E_req.sum(), s_req)[0]
            s_req = s_req.detach()
        ell = median_bandwidth(s_req)
        metrics['ksd_squared'] = float(compute_ksd_squared(s_req, scores.detach(), ell).item())
    except Exception as e:
        metrics['ksd_squared'] = float('nan')

    return metrics


def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)

    # Find latest checkpoint
    ckpt_dir = PROJECT_ROOT / 'results' / 'lj13_as' / 'seed_0' / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('checkpoint_*.pt'),
                   key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else -1)
    # Filter out checkpoint_latest.pt
    numbered = [c for c in ckpts if c.stem.split('_')[1].isdigit()]
    latest_ckpt = numbered[-1] if numbered else ckpt_dir / 'checkpoint_latest.pt'
    ckpt_name = latest_ckpt.name

    print("=" * 70, flush=True)
    print(f"LJ13 AS Evaluation — {ckpt_name}", flush=True)
    print("Metrics: energy_W2, Rg_W2, dist_W2 (fixed √), KSD²", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    # Reference
    ref_path = PROJECT_ROOT / 'data' / 'test_split_LJ13-1000.npy'
    ref_np = np.load(ref_path, allow_pickle=True)
    ref_samples = remove_mean(
        torch.tensor(ref_np, dtype=torch.float32), N_PARTICLES, SPATIAL_DIM
    )
    print(f"Reference: {ref_samples.shape}", flush=True)

    # Energy
    energy_cfg = OmegaConf.create({
        '_target_': 'adjoint_samplers.energies.lennard_jones_energy.LennardJonesEnergy',
        'dim': DIM, 'n_particles': N_PARTICLES,
    })
    energy = hydra.utils.instantiate(energy_cfg, device=DEVICE)

    # Load AS model
    exp_dir = PROJECT_ROOT / 'results' / 'lj13_as' / 'seed_0'
    cfg = OmegaConf.load(exp_dir / 'config.yaml')
    cfg_r = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    source = hydra.utils.instantiate(cfg_r.source, device=DEVICE)
    ref_sde = hydra.utils.instantiate(cfg_r.ref_sde).to(DEVICE)
    controller = hydra.utils.instantiate(cfg_r.controller).to(DEVICE)
    sde = ControlledSDE(ref_sde, controller).to(DEVICE)

    print(f"Loading checkpoint: {latest_ckpt}", flush=True)
    ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    ts_cfg = {
        't0': float(cfg_r.timesteps.t0),
        't1': float(cfg_r.timesteps.t1),
        'steps': int(cfg_r.nfe),
        'rescale_t': cfg_r.get('rescale_t', None),
    }

    seed_results = []
    for seed in range(N_EVAL_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed * 7777)
        np.random.seed(seed * 7777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed * 7777)

        print(f"\n  Seed {seed}:", flush=True)

        x1_list = []
        n = 0
        while n < N_SAMPLES:
            b = min(500, N_SAMPLES - n)
            x0 = source.sample([b]).to(DEVICE)
            ts = train_utils.get_timesteps(**ts_cfg).to(DEVICE)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            x1_list.append(x1)
            n += b
        samples = torch.cat(x1_list)[:N_SAMPLES]
        samples = remove_mean(samples, N_PARTICLES, SPATIAL_DIM)

        m = compute_metrics(samples, energy, ref_samples)
        dt = time.time() - t0

        seed_results.append(m)
        parts = []
        for k in METRIC_KEYS:
            if k in m:
                if k == 'ksd_squared':
                    parts.append(f"KSD²={m[k]:.2f}")
                elif k == 'mean_energy':
                    parts.append(f"<E>={m[k]:.2f}")
                else:
                    parts.append(f"{k}={m[k]:.4f}")
        print(f"    {' | '.join(parts)}  [{dt:.1f}s]", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"AS (LJ13) {ckpt_name} — mean±std", flush=True)
    print(f"{'='*70}", flush=True)
    for key in METRIC_KEYS:
        vals = [m[key] for m in seed_results if key in m and np.isfinite(m[key])]
        if vals:
            print(f"  {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

    out_path = PROJECT_ROOT / 'evaluation' / 'eval_lj13_as_results.json'
    with open(out_path, 'w') as f:
        json.dump(seed_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
