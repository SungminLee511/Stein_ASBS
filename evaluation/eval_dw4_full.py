"""
eval_dw4_full.py — Full DW4 evaluation: energy_W2, eq_W2, dist_W2, KSD², ESS.

Evaluates checkpoint_latest for both ASBS and KSD-ASBS (λ=1.0).
Same setup as RESULTS.md: 2000 samples × 5 sampling seeds (0-4).
ESS: 2000 trajectories per seed (Girsanov importance weights).

Usage:
    cd /home/RESEARCH/Stein_ASBS
    python -u evaluation/eval_dw4_full.py
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

# Import ESS utilities
from evaluation.evaluate_ess import sdeint_with_noise, compute_girsanov_log_weights, compute_ess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
ESS_BATCH_SIZE = 500

# ── Experiments ──
EXPERIMENTS = {
    'ASBS (Baseline)': {
        'dir': PROJECT_ROOT / 'results' / 'dw4_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
    },
    'KSD-ASBS (λ=1.0)': {
        'dir': PROJECT_ROOT / 'results' / 'dw4_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_latest.pt',
    },
}


def load_model(exp_dir, ckpt_name, device):
    """Load model from experiment directory."""
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

    return sde, source, energy, ts_cfg, cfg_r


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device, batch_size=500):
    """Generate terminal samples."""
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
def compute_w2_metrics(samples, energy, ref_samples, n_particles=4, spatial_dim=2):
    """Compute energy_W2, eq_W2, dist_W2."""
    B = samples.shape[0]
    device = samples.device

    # Subsample reference
    idxs = torch.randperm(len(ref_samples))[:B]
    ref = ref_samples[idxs].to(device)

    # energy_W2
    gen_E = energy.eval(samples)
    ref_E = energy.eval(ref)
    energy_w2 = float(pot.emd2_1d(ref_E.cpu().numpy(), gen_E.cpu().numpy()) ** 0.5)

    # dist_W2
    gen_dist = interatomic_dist(samples, n_particles, spatial_dim)
    ref_dist = interatomic_dist(ref, n_particles, spatial_dim)
    dist_w2 = float(pot.emd2_1d(
        gen_dist.cpu().numpy().reshape(-1),
        ref_dist.cpu().numpy().reshape(-1),
    ))

    # eq_W2
    M = dist_point_clouds(
        samples.reshape(-1, n_particles, spatial_dim).cpu(),
        ref.reshape(-1, n_particles, spatial_dim).cpu(),
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


def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 70, flush=True)
    print(f"DW4 Full Evaluation — energy_W2, eq_W2, dist_W2, KSD², ESS", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    # Load reference samples
    ref_path = PROJECT_ROOT / 'data' / 'test_split_DW4.npy'
    ref_np = np.load(ref_path, allow_pickle=True)
    ref_samples = remove_mean(torch.tensor(ref_np, dtype=torch.float32), 4, 2)
    print(f"Reference: {ref_samples.shape}", flush=True)

    all_results = {}

    for method_name, exp_info in EXPERIMENTS.items():
        print(f"\n{'='*70}", flush=True)
        print(f"  {method_name}", flush=True)
        print(f"  Dir: {exp_info['dir']}", flush=True)
        print(f"{'='*70}", flush=True)

        sde, source, energy, ts_cfg, cfg = load_model(exp_info['dir'], exp_info['ckpt'], DEVICE)

        seed_results = []
        for seed in range(N_EVAL_SEEDS):
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            print(f"\n  Seed {seed}:", flush=True)

            # Generate samples
            samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE)

            # W2 metrics
            print(f"    Computing W2 metrics...", flush=True)
            energy_w2, eq_w2, dist_w2 = compute_w2_metrics(samples, energy, ref_samples)

            # KSD²
            print(f"    Computing KSD²...", flush=True)
            ksd2 = compute_ksd(samples, energy)

            # ESS
            print(f"    Computing ESS ({N_SAMPLES} trajectories)...", flush=True)
            torch.manual_seed(seed + 10000)  # different seed for ESS trajectories
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed + 10000)
            ess, ess_frac = compute_ess_for_seed(sde, source, ts_cfg, N_SAMPLES, DEVICE, ESS_BATCH_SIZE)

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

        all_results[method_name] = seed_results

        # Per-method summary
        for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'ess']:
            vals = [r[key] for r in seed_results]
            print(f"    {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)

        del sde, source, energy
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n\n{'='*70}", flush=True)
    print("SUMMARY — DW4 Full Evaluation", flush=True)
    print(f"{'='*70}", flush=True)

    header = f"{'Seed':<6} {'Method':<22} {'energy_W2':>10} {'eq_W2':>10} {'dist_W2':>10} {'KSD²':>10} {'ESS':>10} {'ESS%':>8}"
    print(f"\n{header}", flush=True)
    print("─" * len(header), flush=True)

    for method_name, seed_results in all_results.items():
        for r in seed_results:
            print(
                f"{r['seed']:<6} {method_name:<22} "
                f"{r['energy_w2']:>10.4f} {r['eq_w2']:>10.4f} {r['dist_w2']:>10.6f} "
                f"{r['ksd_squared']:>10.6f} {r['ess']:>10.2f} {r['ess_fraction']*100:>7.4f}%",
                flush=True,
            )
        # Mean ± std
        for key in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'ess']:
            vals = [r[key] for r in seed_results]
            print(f"       {'mean±std':<22} {key}: {np.mean(vals):.6f} ± {np.std(vals):.6f}", flush=True)
        print(flush=True)

    # ── Save ──
    out_path = PROJECT_ROOT / 'evaluation' / 'eval_dw4_full_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
