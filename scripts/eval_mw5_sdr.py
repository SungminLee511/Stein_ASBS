"""
scripts/eval_mw5_sdr.py

Evaluate MW5 (5D Many-Well): Vanilla ASBS vs SDR beta={0.5, 0.7, 1.0}
Metrics: Mode Weight TV, Energy W2, W2, Sinkhorn
(KL excluded — infeasible for 5D grid integration)
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from omegaconf import OmegaConf
import hydra
import ot

from adjoint_samplers.components.sde import ControlledSDE, sdeint
import adjoint_samplers.utils.train_utils as train_utils


# ====================================================================
# Constants
# ====================================================================

RESULTS_DIR = Path('/home/sky/SML/Stein_ASBS/results')
EVAL_DIR = Path('/home/sky/SML/Stein_ASBS/evaluation')


# ====================================================================
# Loading & Sampling
# ====================================================================

def load_model(exp_dir, device, ckpt_override=None):
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir / '.hydra' / 'config.yaml'
    if not cfg_path.exists():
        cfg_path = exp_dir / 'config.yaml'
    if ckpt_override:
        ckpt_path = Path(ckpt_override)
    else:
        ckpt_path = exp_dir / 'checkpoints' / 'checkpoint_latest.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])

    ts_cfg = {
        't0': float(cfg.timesteps.t0),
        't1': float(cfg.timesteps.t1),
        'steps': int(cfg.timesteps.steps),
        'rescale_t': cfg.timesteps.rescale_t if cfg.timesteps.rescale_t is not None else None,
    }
    return sde, source, energy, ts_cfg


@torch.no_grad()
def generate_samples(sde, source, ts_cfg, n_samples, device):
    x0 = source.sample([n_samples]).to(device)
    ts = train_utils.get_timesteps(**ts_cfg).to(device)
    _, x1 = sdeint(sde, x0, ts, only_boundary=True)
    return x1


# ====================================================================
# Metrics
# ====================================================================

def assign_modes(samples, centers, threshold_factor=3.0, std=0.3):
    dists = torch.cdist(samples, centers)
    nearest = dists.argmin(dim=1)
    min_dists = dists.min(dim=1).values
    threshold = threshold_factor * std
    assignments = nearest.clone()
    assignments[min_dists > threshold] = -1
    K = centers.shape[0]
    counts = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        counts[k] = (assignments == k).sum()
    return assignments, counts


def compute_mode_weight_tv(counts):
    K = len(counts)
    total = sum(counts)
    if total == 0:
        return 1.0
    empirical = np.array(counts, dtype=np.float64) / total
    uniform = np.ones(K) / K
    return 0.5 * np.abs(empirical - uniform).sum()


def compute_w2_distance(samples, ref_samples):
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    w2_sq = ot.emd2(wa, wb, M)
    return float(np.sqrt(max(w2_sq, 0)))


def compute_sinkhorn_divergence(samples, ref_samples, reg=0.1):
    a = samples.cpu().numpy()
    b = ref_samples.cpu().numpy()
    n, m = len(a), len(b)
    wa = np.ones(n) / n
    wb = np.ones(m) / m
    M = ot.dist(a, b, metric='sqeuclidean')
    sinkhorn = ot.sinkhorn2(wa, wb, M, reg=reg)
    return float(sinkhorn)


def compute_energy_w2(samples, ref_samples, energy):
    with torch.no_grad():
        E_gen = energy.eval(samples).cpu().numpy()
        E_ref = energy.eval(ref_samples[:len(samples)]).cpu().numpy()
    E_gen_sorted = np.sort(E_gen)
    E_ref_sorted = np.sort(E_ref)
    n = min(len(E_gen_sorted), len(E_ref_sorted))
    w2 = np.sqrt(np.mean((E_gen_sorted[:n] - E_ref_sorted[:n])**2))
    return float(w2)


def compute_all_metrics(samples, energy, centers, std, ref_samples):
    assignments, counts = assign_modes(samples, centers.to(samples.device), std=std)
    counts_list = counts.cpu().tolist()

    metrics = {
        'mode_weight_tv': compute_mode_weight_tv(counts_list),
        'energy_w2': compute_energy_w2(samples, ref_samples, energy),
        'w2': compute_w2_distance(samples, ref_samples),
        'sinkhorn': compute_sinkhorn_divergence(samples, ref_samples),
    }
    return metrics


# ====================================================================
# Main
# ====================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 2000

    # Define methods and their experiment directories
    methods = {
        'ASBS': {
            'seeds': [
                RESULTS_DIR / 'mw5_asbs' / 'seed_0',
                RESULTS_DIR / 'mw5_asbs' / 'seed_1',
                RESULTS_DIR / 'mw5_asbs' / 'seed_2',
            ],
        },
        r'SDR $\beta$=0.5': {
            'seeds': [
                RESULTS_DIR / 'mw5_sdr_b0.5_s0' / 'seed_0',
                RESULTS_DIR / 'mw5_sdr_b0.5_s1' / 'seed_1',
                RESULTS_DIR / 'mw5_sdr_b0.5_s2' / 'seed_2',
            ],
        },
        r'SDR $\beta$=0.7': {
            'seeds': [
                RESULTS_DIR / 'mw5_sdr_b0.7_s0' / 'seed_0',
                RESULTS_DIR / 'mw5_sdr_b0.7_s1' / 'seed_1',
                RESULTS_DIR / 'mw5_sdr_b0.7_s2' / 'seed_2',
            ],
        },
        r'SDR $\beta$=1.0': {
            'seeds': [
                RESULTS_DIR / 'mw5_sdr_b1.0_s0' / 'seed_0',
                RESULTS_DIR / 'mw5_sdr_b1.0_s1' / 'seed_1',
                RESULTS_DIR / 'mw5_sdr_b1.0_s2' / 'seed_2',
            ],
        },
    }

    print("=" * 70)
    print("  MW5 (5D Many-Well) — SDR Evaluation")
    print("=" * 70)

    # Load energy from first available model
    first_dir = methods['ASBS']['seeds'][0]
    _, _, energy, _ = load_model(first_dir, device)
    centers = energy.get_mode_centers().to(device)
    std = 0.3  # approximate well width for mode assignment
    ref_samples = energy.get_ref_samples(5000).to(device)

    # ---- Compute metrics per method (mean +/- std over seeds) ----
    all_results = {}

    for method_name, method_info in methods.items():
        print(f"\n  === {method_name} ===")
        seed_metrics = []

        for seed_dir in method_info['seeds']:
            print(f"    Seed: {seed_dir.name} ({seed_dir.parent.name})")
            sde, source, _, ts_cfg = load_model(seed_dir, device)

            torch.manual_seed(0)
            samples = generate_samples(sde, source, ts_cfg, n_samples, device)
            m = compute_all_metrics(samples, energy, centers, std, ref_samples)
            seed_metrics.append(m)
            print(f"      Mode TV={m['mode_weight_tv']:.4f}  E-W2={m['energy_w2']:.4f}  W2={m['w2']:.4f}  Sinkhorn={m['sinkhorn']:.4f}")

            del sde, source, samples
            torch.cuda.empty_cache()

        # Aggregate: mean +/- std
        keys = seed_metrics[0].keys()
        agg = {}
        for k in keys:
            vals = [m[k] for m in seed_metrics if not (isinstance(m[k], float) and np.isnan(m[k]))]
            if len(vals) > 0:
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))
            else:
                agg[f'{k}_mean'] = float('nan')
                agg[f'{k}_std'] = float('nan')
        all_results[method_name] = agg

    # ---- Print results table ----
    print("\n" + "=" * 100)
    metric_keys = ['mode_weight_tv', 'energy_w2', 'w2', 'sinkhorn']
    metric_labels = {
        'mode_weight_tv': 'Mode Weight TV ↓',
        'energy_w2': 'Energy W2 ↓',
        'w2': 'W2 Distance ↓',
        'sinkhorn': 'Sinkhorn Div ↓',
    }

    method_names = list(methods.keys())
    header = f"  {'Metric':<35}" + "".join(f"{m:>22}" for m in method_names)
    print(header)
    print("-" * 100)

    for k in metric_keys:
        row = f"  {metric_labels[k]:<35}"
        for m_name in method_names:
            mean_val = all_results[m_name][f'{k}_mean']
            std_val = all_results[m_name][f'{k}_std']
            if np.isnan(mean_val):
                row += f"{'N/A':>22}"
            else:
                row += f"{mean_val:>12.4f}±{std_val:<8.4f}"
        print(row)
    print("=" * 100)

    # ---- Save metrics JSON ----
    json_path = EVAL_DIR / 'mw5_sdr_results.json'
    json_results = {}
    for m_name, agg in all_results.items():
        json_results[m_name] = {k: v for k, v in agg.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved metrics: {json_path}")

    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')
    print(f"\n=== Evaluation complete at {now_kst} ===")


if __name__ == '__main__':
    main()
