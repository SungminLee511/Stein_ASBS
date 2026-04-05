"""
eval_imq_ablation.py — Evaluate all 4 RotGMM IMQ-KSD-ASBS experiments.
Evaluates: rotgmm{10,30,50,100}_imq_asbs
5 eval seeds, 2000 samples each. Saves per-dimension JSON alongside existing results.
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

EXPERIMENTS = {
    "rotgmm10_imq_asbs": {
        "run_dir": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm10_imq_asbs/seed_0"),
        "results_dir": Path("/home/RESEARCH/Stein_ASBS/evaluation/results_rotgmm10"),
        "dim": 10,
    },
    "rotgmm30_imq_asbs": {
        "run_dir": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm30_imq_asbs/seed_0"),
        "results_dir": Path("/home/RESEARCH/Stein_ASBS/evaluation/results_rotgmm30"),
        "dim": 30,
    },
    "rotgmm50_imq_asbs": {
        "run_dir": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm50_imq_asbs/seed_0"),
        "results_dir": Path("/home/RESEARCH/Stein_ASBS/evaluation/results_rotgmm50"),
        "dim": 50,
    },
    "rotgmm100_imq_asbs": {
        "run_dir": Path("/home/RESEARCH/Stein_ASBS/results/rotgmm100_imq_asbs/seed_0"),
        "results_dir": Path("/home/RESEARCH/Stein_ASBS/evaluation/results_rotgmm100"),
        "dim": 100,
    },
}


def load_and_generate(run_dir, seed, n_samples):
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


def count_modes_nearest(samples, energy):
    centers = energy.get_mode_centers().to(samples.device)
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)
    nearest_counts = [(nearest == k).sum().item() for k in range(centers.shape[0])]
    min_dists = dists.min(dim=0).values
    return {
        'nearest_mode_counts': nearest_counts,
        'n_modes_nearest': sum(1 for c in nearest_counts if c > 0),
        'n_modes_total': int(centers.shape[0]),
        'min_dists_to_centers': min_dists.cpu().tolist(),
    }


@torch.no_grad()
def compute_metrics(samples, energy, ref_samples=None):
    metrics = {}
    N, D = samples.shape

    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

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

    cov = count_modes_nearest(samples, energy)
    metrics['nearest_mode_counts'] = cov['nearest_mode_counts']
    metrics['n_modes_nearest'] = cov['n_modes_nearest']
    metrics['n_modes_total'] = cov['n_modes_total']
    metrics['min_dists_to_centers'] = cov['min_dists_to_centers']

    return metrics


def evaluate_experiment(exp_name, info):
    run_dir = info["run_dir"]
    results_dir = info["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name} (d={info['dim']})")
    print(f"  run_dir: {run_dir}")
    print(f"{'='*60}")

    seed_metrics = []
    for eval_seed in range(N_EVAL_SEEDS):
        print(f"  Seed {eval_seed}...")
        samples, energy = load_and_generate(run_dir, eval_seed * 7777, N_SAMPLES)

        ref_samples = None
        if hasattr(energy, 'get_ref_samples'):
            ref_samples = energy.get_ref_samples()

        m = compute_metrics(samples, energy, ref_samples)
        seed_metrics.append(m)

        print(f"    energy_w2={m.get('energy_w2', 'N/A'):.4f}, "
              f"ksd²={m['ksd_squared']:.4f}, "
              f"modes={m['n_modes_nearest']}/{m['n_modes_total']}")
        print(f"    nearest_counts={m['nearest_mode_counts']}")

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
            arr = np.array(vals)
            agg[key] = {
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
                'per_seed': vals,
            }

    # Summary
    print(f"\n  Summary for {exp_name}:")
    for k in ['mean_energy', 'energy_w2', 'ksd_squared', 'n_modes_nearest']:
        if k in agg:
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    return agg


def main():
    all_imq_results = {}

    for exp_name, info in EXPERIMENTS.items():
        agg = evaluate_experiment(exp_name, info)
        all_imq_results[exp_name] = agg

        # Also merge into the existing per-dimension JSON
        dim_label = f"rotgmm{info['dim']}"
        json_path = info["results_dir"] / f"{dim_label}_results.json"

        existing = {}
        if json_path.exists():
            with open(json_path, 'r') as f:
                existing = json.load(f)

        existing[exp_name] = agg
        with open(json_path, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"  Updated {json_path}")

    # Save combined IMQ results
    combined_path = Path("/home/RESEARCH/Stein_ASBS/evaluation/imq_ablation_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_imq_results, f, indent=2, default=str)
    print(f"\nSaved combined IMQ results to {combined_path}")

    # Print final comparison table
    print(f"\n{'='*70}")
    print("IMQ ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':30s} | {'energy_W2':>18s} | {'KSD²':>18s} | {'Modes':>6s}")
    print("-" * 78)
    for exp_name, agg in all_imq_results.items():
        ew2 = agg.get('energy_w2', {})
        ksd = agg.get('ksd_squared', {})
        modes = agg.get('n_modes_nearest', {})
        print(f"{exp_name:30s} | {ew2.get('mean',0):>8.4f}±{ew2.get('std',0):<8.4f} | "
              f"{ksd.get('mean',0):>8.4f}±{ksd.get('std',0):<8.4f} | "
              f"{modes.get('mean',0):>4.1f}/8")


if __name__ == "__main__":
    main()
