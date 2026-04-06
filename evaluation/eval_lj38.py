"""
eval_lj38.py — Evaluate LJ38 checkpoint_100 for three experiments:
  lj38_asbs (baseline), lj38_imq_asbs (IMQ-KSD), lj38_ksd_asbs (RBF-KSD)

Generates samples from each checkpoint, computes:
  - Energy stats (mean, std, min, max)
  - KSD²
  - Energy W2 (vs reference)
  - Interatomic distance W2
  - EQ W2 (Earth Mover on point clouds)

Results saved to evaluation/lj38_eval_results.json
"""

import json
import sys
import time
import torch
import numpy as np
from pathlib import Path

import ot as pot

# --- Setup path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
import hydra

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist, dist_point_clouds
from adjoint_samplers.utils.graph_utils import remove_mean
import adjoint_samplers.utils.train_utils as train_utils


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
BATCH_SIZE = 500  # conservative to avoid OOM alongside training

EXPERIMENTS = {
    'lj38_asbs': {
        'dir': PROJECT_ROOT / 'results' / 'lj38_asbs' / 'seed_0',
        'ckpt': 'checkpoint_100.pt',
    },
    'lj38_imq_asbs': {
        'dir': PROJECT_ROOT / 'results' / 'lj38_imq_asbs' / 'seed_0',
        'ckpt': 'checkpoint_100.pt',
    },
    'lj38_ksd_asbs': {
        'dir': PROJECT_ROOT / 'results' / 'lj38_ksd_asbs' / 'seed_0',
        'ckpt': 'checkpoint_100.pt',
    },
}

# LJ38 constants
N_PARTICLES = 38
SPATIAL_DIM = 3
DIM = 114
NFE = 1000
SIGMA_MAX = 2
SIGMA_MIN = 0.001
SCALE = 1.5
MAX_GRAD_E_NORM = 100

REF_PATH = PROJECT_ROOT / 'data' / 'test_split_LJ38-1000.npy'


def load_experiment(exp_name, exp_info):
    """Load model from checkpoint."""
    exp_dir = exp_info['dir']
    ckpt_path = exp_dir / 'checkpoints' / exp_info['ckpt']
    cfg_path = exp_dir / 'config.yaml'

    if not ckpt_path.exists():
        print(f"  ERROR: {ckpt_path} not found")
        return None

    # Load config to get matcher type, but we'll build components manually
    # since config has unresolved ${} variables
    cfg = OmegaConf.load(cfg_path)

    # Build energy
    energy = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.energies.lennard_jones_energy.LennardJonesEnergy',
            'dim': DIM, 'n_particles': N_PARTICLES,
        }),
        device=DEVICE,
    )

    # Build source
    source = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.utils.dist_utils.CenteredParticlesHarmonic',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM, 'scale': SCALE,
        }),
        device=DEVICE,
    )

    # Build ref SDE
    ref_sde = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.components.sde.GraphVESDE',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM,
            'sigma_max': SIGMA_MAX, 'sigma_min': SIGMA_MIN,
        })
    ).to(DEVICE)

    # Build controller (EGNN)
    controller = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.components.model.EGNN_dynamics',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM,
            'hidden_nf': 128, 'n_layers': 5,
            'act_fn': {'_target_': 'torch.nn.SiLU'},
            'recurrent': True, 'tanh': True, 'attention': True,
            'condition_time': True, 'agg': 'sum',
        })
    ).to(DEVICE)

    sde = ControlledSDE(ref_sde, controller).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    print(f"  Loaded {exp_name} from {exp_info['ckpt']} (epoch {ckpt.get('epoch', '?')})")

    return {
        'sde': sde, 'source': source, 'energy': energy,
        'controller': controller,
    }


@torch.no_grad()
def generate_samples(sde, source, n_samples, batch_size):
    """Generate terminal samples in batches."""
    ts = train_utils.get_timesteps(t0=0.0, t1=1.0, steps=NFE, rescale_t=None).to(DEVICE)
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(DEVICE)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1.cpu())
        n += b
    return torch.cat(x1_list)[:n_samples].to(DEVICE)


@torch.no_grad()
def compute_metrics(samples, energy, ref_samples):
    """Compute all metrics for a set of samples."""
    metrics = {}
    N, D = samples.shape

    # Energy stats
    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

    # KSD²
    with torch.enable_grad():
        s = samples.detach().requires_grad_(True)
        E = energy.eval(s)
        scores = -torch.autograd.grad(E.sum(), s)[0]
        s = s.detach()

    ell = median_bandwidth(samples)
    N_ksd = min(N, 2000)
    idx = torch.randperm(N, device=samples.device)[:N_ksd]
    metrics['ksd_squared'] = compute_ksd_squared(
        samples[idx], scores[idx].detach(), ell
    ).item()
    metrics['bandwidth'] = ell.item()

    # Reference-based
    if ref_samples is not None:
        ref = ref_samples.to(samples.device)
        B = min(N, len(ref))
        idx_g = torch.randperm(N, device=samples.device)[:B]
        idx_r = torch.randperm(len(ref), device=samples.device)[:B]
        gen = samples[idx_g]
        ref_sub = ref[idx_r]

        ref_E = energy.eval(ref_sub)
        metrics['ref_mean_energy'] = ref_E.mean().item()
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E.cpu().numpy(), gen_E[idx_g].cpu().numpy()) ** 0.5
        )

        # Interatomic distance W2
        gen_dist = interatomic_dist(gen, N_PARTICLES, SPATIAL_DIM)
        ref_dist = interatomic_dist(ref_sub, N_PARTICLES, SPATIAL_DIM)
        metrics['dist_w2'] = float(pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        ))

        # EQ W2 (point cloud matching)
        M = dist_point_clouds(
            gen.reshape(-1, N_PARTICLES, SPATIAL_DIM).cpu(),
            ref_sub.reshape(-1, N_PARTICLES, SPATIAL_DIM).cpu(),
        )
        a = torch.ones(M.shape[0]) / M.shape[0]
        b = torch.ones(M.shape[1]) / M.shape[1]
        metrics['eq_w2'] = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)

    return metrics


def main():
    print(f"=== LJ38 Evaluation (3 experiments × {N_EVAL_SEEDS} seeds) ===")
    print(f"Device: {DEVICE}")
    print(f"N_samples: {N_SAMPLES}, Batch size: {BATCH_SIZE}")
    print()

    # Load reference samples
    ref_np = np.load(REF_PATH, allow_pickle=True)
    ref_samples = torch.tensor(ref_np, dtype=torch.float32)
    ref_samples = remove_mean(ref_samples, N_PARTICLES, SPATIAL_DIM)
    print(f"Reference samples: {ref_samples.shape}")

    all_results = {}

    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        exp = load_experiment(exp_name, exp_info)
        if exp is None:
            continue

        seed_metrics = []
        for eval_seed in range(N_EVAL_SEEDS):
            torch.manual_seed(eval_seed * 7777)
            t0 = time.time()
            samples = generate_samples(exp['sde'], exp['source'], N_SAMPLES, BATCH_SIZE)
            gen_time = time.time() - t0

            t0 = time.time()
            m = compute_metrics(samples, exp['energy'], ref_samples)
            eval_time = time.time() - t0
            m['gen_time'] = gen_time
            m['eval_time'] = eval_time

            seed_metrics.append(m)
            print(f"  seed={eval_seed}: E={m['mean_energy']:.2f}±{m['std_energy']:.2f}, "
                  f"KSD²={m['ksd_squared']:.4f}, EW2={m['energy_w2']:.4f}, "
                  f"DW2={m['dist_w2']:.6f}, EQW2={m['eq_w2']:.4f} "
                  f"({gen_time:.1f}s gen, {eval_time:.1f}s eval)")

        # Aggregate
        agg = {}
        for key in seed_metrics[0]:
            vals = [m[key] for m in seed_metrics]
            agg[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }

        all_results[exp_name] = agg

        # Print summary
        print(f"\n  --- {exp_name} Summary ---")
        for k in ['mean_energy', 'ksd_squared', 'energy_w2', 'dist_w2', 'eq_w2']:
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    # Save
    out_path = Path(__file__).parent / 'lj38_eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== Results saved to {out_path} ===")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Metric':<20} {'ASBS (baseline)':<25} {'IMQ-KSD':<25} {'RBF-KSD':<25}")
    print(f"{'='*80}")
    for k in ['mean_energy', 'ksd_squared', 'energy_w2', 'dist_w2', 'eq_w2']:
        row = f"{k:<20}"
        for exp_name in ['lj38_asbs', 'lj38_imq_asbs', 'lj38_ksd_asbs']:
            if exp_name in all_results:
                v = all_results[exp_name][k]
                row += f" {v['mean']:>10.4f} ± {v['std']:<8.4f}"
            else:
                row += f" {'N/A':>20}"
        print(row)
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
