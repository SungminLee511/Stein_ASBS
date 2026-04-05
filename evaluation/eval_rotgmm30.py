"""
eval_rotgmm30.py — Evaluate RotGMM d=30 baseline vs KSD-ASBS.

Computes: energy stats, KSD², energy_W2, mode coverage.
5 eval seeds, 2000 samples each.
Saves JSON + prints summary.
"""

import sys
sys.path.insert(0, "/home/RESEARCH/Stein_ASBS")

import json
import torch
import numpy as np
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import ot as pot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.train_utils import get_timesteps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
RESULTS_DIR = Path("/home/RESEARCH/Stein_ASBS/evaluation/results_rotgmm30")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "rotgmm30_asbs": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm30_asbs/seed_0"),
    "rotgmm30_ksd_asbs": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm30_ksd_asbs/seed_0"),
}


def load_and_generate(run_dir, seed, n_samples):
    """Load checkpoint and generate samples."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    energy = hydra.utils.instantiate(cfg.energy, device=DEVICE)
    source = hydra.utils.instantiate(cfg.source, device=DEVICE)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(DEVICE)
    controller = hydra.utils.instantiate(cfg.controller).to(DEVICE)
    sde = ControlledSDE(ref_sde, controller).to(DEVICE)

    ckpt = torch.load(run_dir / "checkpoints" / "checkpoint_latest.pt",
                      map_location=DEVICE, weights_only=False)
    controller.load_state_dict(ckpt["controller"])
    controller.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    ts_cfg = {"t0": 0.0, "t1": 1.0,
              "steps": cfg.get("nfe", 200),
              "rescale_t": cfg.get("rescale_t", None)}

    x1_list = []
    n = 0
    batch_size = min(n_samples, cfg.get("eval_batch_size", 2000))
    with torch.no_grad():
        while n < n_samples:
            b = min(batch_size, n_samples - n)
            x0 = source.sample([b]).to(DEVICE)
            ts = get_timesteps(**ts_cfg).to(DEVICE)
            _, x1 = sdeint(sde, x0, ts, only_boundary=True)
            x1_list.append(x1)
            n += b

    samples = torch.cat(x1_list)[:n_samples]
    return samples, energy


@torch.no_grad()
def compute_metrics(samples, energy, ref_samples=None):
    """Compute metrics for a set of samples."""
    metrics = {}
    N, D = samples.shape

    # Energy statistics
    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

    # KSD²
    with torch.enable_grad():
        samples_req = samples.detach().requires_grad_(True)
        E = energy.eval(samples_req)
        scores = -torch.autograd.grad(E.sum(), samples_req)[0]
        samples_req = samples_req.detach()

    ell = median_bandwidth(samples)
    N_ksd = min(N, 2000)
    idx = torch.randperm(N, device=samples.device)[:N_ksd]
    metrics['ksd_squared'] = compute_ksd_squared(
        samples[idx], scores[idx].detach(), ell
    ).item()
    metrics['bandwidth'] = ell.item()

    # Reference-based: energy_W2
    if ref_samples is not None:
        ref_samples = ref_samples.to(samples.device)
        B = min(N, len(ref_samples))
        idx_g = torch.randperm(N, device=samples.device)[:B]
        idx_r = torch.randperm(len(ref_samples), device=samples.device)[:B]
        ref_E = energy.eval(ref_samples[idx_r])
        metrics['ref_mean_energy'] = ref_E.mean().item()
        metrics['energy_w2'] = float(
            pot.emd2_1d(ref_E.cpu().numpy(), gen_E[idx_g].cpu().numpy()) ** 0.5
        )

    # Mode coverage
    if hasattr(energy, 'count_modes_covered'):
        cov = energy.count_modes_covered(samples)
        metrics['n_modes_covered'] = cov['n_modes_covered']
        metrics['n_modes_total'] = cov['n_modes_total']
        metrics['coverage_fraction'] = cov['coverage_fraction']
        metrics['per_mode_counts'] = cov['per_mode_counts']

    return metrics


def main():
    all_results = {}

    for exp_name, run_dir in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        seed_metrics = []
        for eval_seed in range(N_EVAL_SEEDS):
            print(f"  Seed {eval_seed}...")
            samples, energy = load_and_generate(run_dir, eval_seed * 7777, N_SAMPLES)

            # Get ref samples
            ref_samples = None
            if hasattr(energy, 'get_ref_samples'):
                ref_samples = energy.get_ref_samples()

            m = compute_metrics(samples, energy, ref_samples)
            seed_metrics.append(m)

            print(f"    energy_w2={m.get('energy_w2', 'N/A'):.4f}, "
                  f"ksd²={m['ksd_squared']:.4f}, "
                  f"modes={m.get('n_modes_covered', '?')}/{m.get('n_modes_total', '?')}")

        # Aggregate
        agg = {}
        for key in seed_metrics[0]:
            vals = [m[key] for m in seed_metrics]
            if isinstance(vals[0], (int, float)):
                agg[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': vals,
                }
            elif isinstance(vals[0], list):
                agg[key] = vals[0]  # Take first seed's list

        all_results[exp_name] = agg

        print(f"\n  Summary for {exp_name}:")
        for k in ['mean_energy', 'energy_w2', 'ksd_squared', 'n_modes_covered', 'coverage_fraction']:
            if k in agg:
                print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    # Save
    out_path = RESULTS_DIR / "rotgmm30_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    bl = all_results.get("rotgmm30_asbs", {})
    ksd = all_results.get("rotgmm30_ksd_asbs", {})

    headers = ['Metric', 'Baseline (mean±std)', 'KSD-ASBS (mean±std)']
    for metric in ['mean_energy', 'energy_w2', 'ksd_squared', 'n_modes_covered', 'coverage_fraction']:
        bl_val = bl.get(metric, {})
        ksd_val = ksd.get(metric, {})
        if bl_val and ksd_val:
            print(f"  {metric:25s}: {bl_val['mean']:.4f}±{bl_val['std']:.4f}  |  {ksd_val['mean']:.4f}±{ksd_val['std']:.4f}")

    # Mode counts (from first seed)
    bl_modes = bl.get('per_mode_counts', [])
    ksd_modes = ksd.get('per_mode_counts', [])
    print(f"\n  Baseline per-mode counts:  {bl_modes}")
    print(f"  KSD-ASBS per-mode counts:  {ksd_modes}")


if __name__ == "__main__":
    main()
