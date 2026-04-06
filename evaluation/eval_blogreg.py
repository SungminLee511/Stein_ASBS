"""
eval_blogreg.py — Evaluate Bayesian Logistic Regression experiments.
Australian (d=15) and German (d=25), baseline vs KSD-ASBS.
5 eval seeds, 2000 samples each. Metrics: energy_W2, KSD², marginal W2.
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
RESULTS_DIR = Path("/home/RESEARCH/Stein_ASBS/evaluation/results_blogreg")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    # Australian dataset (d=15)
    "blogreg_au_asbs": Path("/home/RESEARCH/Stein_ASBS/results/blogreg_au_asbs/seed_0"),
    "blogreg_au_ksd_asbs": Path("/home/RESEARCH/Stein_ASBS/results/blogreg_au_ksd_asbs/seed_0"),
    # German dataset (d=25)
    "blogreg_ge_asbs": Path("/home/RESEARCH/Stein_ASBS/results/blogreg_ge_asbs/seed_0"),
    "blogreg_ge_ksd_asbs": Path("/home/RESEARCH/Stein_ASBS/results/blogreg_ge_ksd_asbs/seed_0"),
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
              "steps": cfg.get("nfe", 100),
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
def compute_metrics(samples, energy, ref_samples):
    """Compute all metrics for a set of generated samples."""
    metrics = {}
    N, D = samples.shape

    # --- Energy statistics ---
    gen_E = energy.eval(samples)
    metrics['mean_energy'] = gen_E.mean().item()
    metrics['std_energy'] = gen_E.std().item()
    metrics['min_energy'] = gen_E.min().item()
    metrics['max_energy'] = gen_E.max().item()

    # --- KSD² ---
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

    # --- Energy W2 vs HMC reference ---
    ref_samples = ref_samples.to(samples.device)
    ref_E = energy.eval(ref_samples)
    metrics['ref_mean_energy'] = ref_E.mean().item()
    metrics['energy_w2'] = float(
        pot.emd2_1d(ref_E.cpu().numpy(), gen_E.cpu().numpy()) ** 0.5
    )

    # --- Marginal W2 (per-dimension mean) ---
    # Compare marginal distributions along each coordinate
    marginal_w2s = []
    n_compare = min(N, len(ref_samples))
    for d in range(D):
        w2_d = pot.emd2_1d(
            ref_samples[:n_compare, d].cpu().numpy(),
            samples[:n_compare, d].cpu().numpy()
        ) ** 0.5
        marginal_w2s.append(float(w2_d))
    metrics['marginal_w2_mean'] = float(np.mean(marginal_w2s))
    metrics['marginal_w2_max'] = float(np.max(marginal_w2s))
    metrics['marginal_w2_per_dim'] = marginal_w2s

    # --- Sample covariance Frobenius error ---
    gen_cov = torch.cov(samples.T)  # (D, D)
    ref_cov = torch.cov(ref_samples.to(samples.device).T)
    metrics['cov_frob_error'] = (gen_cov - ref_cov).norm().item()

    # --- Mean error (L2) ---
    gen_mean = samples.mean(dim=0)
    ref_mean = ref_samples.to(samples.device).mean(dim=0)
    metrics['mean_l2_error'] = (gen_mean - ref_mean).norm().item()

    return metrics


def main():
    # Group experiments by dataset
    datasets = {
        'australian': ['blogreg_au_asbs', 'blogreg_au_ksd_asbs'],
        'german': ['blogreg_ge_asbs', 'blogreg_ge_ksd_asbs'],
    }

    all_results = {}

    for dataset_name, exp_names in datasets.items():
        print(f"\n{'#'*70}")
        print(f"# Dataset: {dataset_name.upper()}")
        print(f"{'#'*70}")

        # Generate HMC reference once per dataset (expensive)
        ref_samples = None

        for exp_name in exp_names:
            run_dir = EXPERIMENTS[exp_name]
            if not run_dir.exists():
                print(f"\n  SKIP {exp_name}: {run_dir} does not exist")
                continue

            print(f"\n{'='*60}")
            print(f"Evaluating: {exp_name}")
            print(f"{'='*60}")

            seed_metrics = []
            for eval_seed in range(N_EVAL_SEEDS):
                print(f"  Seed {eval_seed}...")
                samples, energy = load_and_generate(run_dir, eval_seed * 7777, N_SAMPLES)

                # Generate HMC reference on first use (shared across baseline & KSD for same dataset)
                if ref_samples is None:
                    print(f"\n  Generating HMC reference samples for {dataset_name}...")
                    ref_samples = energy.get_ref_samples()
                    print(f"  HMC reference: {ref_samples.shape[0]} samples, "
                          f"mean energy = {energy.eval(ref_samples.to(samples.device)).mean().item():.4f}")

                m = compute_metrics(samples, energy, ref_samples)
                seed_metrics.append(m)

                print(f"    energy_w2={m['energy_w2']:.4f}, "
                      f"ksd²={m['ksd_squared']:.6f}, "
                      f"marginal_w2={m['marginal_w2_mean']:.4f}, "
                      f"cov_frob={m['cov_frob_error']:.4f}")

            # Aggregate across seeds
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

            all_results[exp_name] = agg

            print(f"\n  Summary for {exp_name}:")
            for k in ['mean_energy', 'energy_w2', 'ksd_squared', 'marginal_w2_mean',
                       'cov_frob_error', 'mean_l2_error']:
                if k in agg:
                    print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    # Save results JSON
    out_path = RESULTS_DIR / "blogreg_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # ====== Comparison tables ======
    for dataset_name, (bl_name, ksd_name) in [
        ('Australian (d=15)', ('blogreg_au_asbs', 'blogreg_au_ksd_asbs')),
        ('German (d=25)', ('blogreg_ge_asbs', 'blogreg_ge_ksd_asbs')),
    ]:
        bl = all_results.get(bl_name, {})
        ksd = all_results.get(ksd_name, {})
        if not bl or not ksd:
            continue

        print(f"\n{'='*70}")
        print(f"COMPARISON: {dataset_name}")
        print(f"{'='*70}")
        print(f"  {'Metric':25s}  {'Baseline':>20s}  {'KSD-ASBS':>20s}  {'Δ':>10s}")
        print(f"  {'-'*25}  {'-'*20}  {'-'*20}  {'-'*10}")

        for metric in ['energy_w2', 'ksd_squared', 'mean_energy', 'marginal_w2_mean',
                        'cov_frob_error', 'mean_l2_error']:
            bl_val = bl.get(metric, {})
            ksd_val = ksd.get(metric, {})
            if bl_val and ksd_val:
                bl_m, bl_s = bl_val['mean'], bl_val['std']
                ksd_m, ksd_s = ksd_val['mean'], ksd_val['std']
                if bl_m != 0:
                    delta = (bl_m - ksd_m) / abs(bl_m) * 100
                    sign = "↓" if delta > 0 else "↑"
                    delta_str = f"{abs(delta):.1f}% {sign}"
                else:
                    delta_str = "N/A"
                print(f"  {metric:25s}  {bl_m:>8.4f}±{bl_s:<8.4f}  {ksd_m:>8.4f}±{ksd_s:<8.4f}  {delta_str:>10s}")


if __name__ == "__main__":
    main()
