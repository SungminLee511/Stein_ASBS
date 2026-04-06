"""
eval_dw4_lambda_ablation.py — DW4 KSD λ-ablation evaluation.

Evaluates baseline ASBS + KSD-ASBS at λ ∈ {0.1, 0.5, 1.0, 5.0, 10.0}.
The λ=1.0 result is taken from the existing eval_results_dw4.json.

Usage:
    python eval_dw4_lambda_ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import hydra
from omegaconf import OmegaConf

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.utils.train_utils import get_timesteps

cudnn.benchmark = True

EVAL_DIR = Path(__file__).parent
REPO_ROOT = EVAL_DIR.parent

# ── Directories ──
BASELINE_DIR = REPO_ROOT / "baselines" / "dw4_asbs"
LAMBDA_DIRS = {
    0.1:  REPO_ROOT / "results" / "dw4_ksd_l0.1"  / "seed_0",
    0.5:  REPO_ROOT / "results" / "dw4_ksd_l0.5"  / "seed_0",
    5.0:  REPO_ROOT / "results" / "dw4_ksd_l5.0"  / "seed_0",
    10.0: REPO_ROOT / "results" / "dw4_ksd_l10.0" / "seed_0",
}

# λ=1.0 was the original KSD-ASBS run
EXISTING_RESULTS_PATH = EVAL_DIR / "eval_results_dw4.json"

SEEDS = [0, 1, 2, 3, 4]
NUM_EVAL_SAMPLES = 2000
OUTPUT_JSON = EVAL_DIR / "eval_results_dw4_lambda_ablation.json"


def load_model_from_dir(run_dir: str, device: str = "cuda"):
    """Load controller + corrector + SDE from a run directory."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "checkpoint_latest.pt"

    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    cfg = OmegaConf.load(cfg_path)

    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(checkpoint["controller"])

    evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)
    return cfg, sde, source, controller, evaluator


@torch.no_grad()
def generate_and_evaluate(sde, source, cfg, evaluator, seed, num_eval_samples, device="cuda"):
    """Generate samples with a given seed and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_gen = 0
    x1_list = []
    eval_batch_size = cfg.eval_batch_size

    while n_gen < num_eval_samples:
        B = min(eval_batch_size, num_eval_samples - n_gen)
        x0 = source.sample([B,]).to(device)
        timesteps = get_timesteps(**cfg.timesteps).to(x0)
        x0, x1 = sdeint(sde, x0, timesteps, only_boundary=True)
        x1_list.append(x1)
        n_gen += x1.shape[0]

    samples = torch.cat(x1_list, dim=0)
    eval_dict = evaluator(samples)

    return {
        "energy_w2": float(eval_dict["energy_w2"]),
        "eq_w2": float(eval_dict["eq_w2"]),
        "dist_w2": float(eval_dict["dist_w2"]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = {}

    # ── 1. Baseline ──
    print("\n" + "=" * 70)
    print("  Loading BASELINE (no KSD)")
    print("=" * 70)
    bl_cfg, bl_sde, bl_source, bl_ctrl, bl_eval = load_model_from_dir(BASELINE_DIR, device)
    bl_ctrl.eval()

    all_results["baseline"] = []
    for seed in SEEDS:
        metrics = generate_and_evaluate(bl_sde, bl_source, bl_cfg, bl_eval, seed, NUM_EVAL_SAMPLES, device)
        print(f"  seed={seed}  energy_W2={metrics['energy_w2']:.4f}  eq_W2={metrics['eq_w2']:.4f}  dist_W2={metrics['dist_w2']:.6f}")
        all_results["baseline"].append({"seed": seed, **metrics})

    # Free GPU memory
    del bl_sde, bl_source, bl_ctrl, bl_eval
    torch.cuda.empty_cache()

    # ── 2. λ=1.0 from existing results ──
    if EXISTING_RESULTS_PATH.exists():
        print("\n" + "=" * 70)
        print("  λ=1.0 — Loading from existing eval_results_dw4.json")
        print("=" * 70)
        with open(EXISTING_RESULTS_PATH) as f:
            existing = json.load(f)
        all_results["lambda_1.0"] = existing["results"]["ksd_asbs"]
        for r in all_results["lambda_1.0"]:
            print(f"  seed={r['seed']}  energy_W2={r['energy_w2']:.4f}  eq_W2={r['eq_w2']:.4f}  dist_W2={r['dist_w2']:.6f}")
    else:
        print(f"\n  WARNING: {EXISTING_RESULTS_PATH} not found, skipping λ=1.0")

    # ── 3. Evaluate each λ ──
    for lam, run_dir in sorted(LAMBDA_DIRS.items()):
        print("\n" + "=" * 70)
        print(f"  Loading KSD-ASBS  λ={lam}")
        print("=" * 70)

        cfg, sde, source, ctrl, evaluator = load_model_from_dir(run_dir, device)
        ctrl.eval()

        key = f"lambda_{lam}"
        all_results[key] = []
        for seed in SEEDS:
            metrics = generate_and_evaluate(sde, source, cfg, evaluator, seed, NUM_EVAL_SAMPLES, device)
            print(f"  seed={seed}  energy_W2={metrics['energy_w2']:.4f}  eq_W2={metrics['eq_w2']:.4f}  dist_W2={metrics['dist_w2']:.6f}")
            all_results[key].append({"seed": seed, **metrics})

        del sde, source, ctrl, evaluator
        torch.cuda.empty_cache()

    # ── 4. Summary table ──
    print("\n" + "=" * 70)
    print("  SUMMARY — DW4 λ-Ablation")
    print("=" * 70)
    print(f"\n  {'Method':<18} {'energy_W2':>14} {'eq_W2':>14} {'dist_W2':>14}")
    print(f"  {'─' * 18} {'─' * 14} {'─' * 14} {'─' * 14}")

    for key in ["baseline", "lambda_0.1", "lambda_0.5", "lambda_1.0", "lambda_5.0", "lambda_10.0"]:
        if key not in all_results:
            continue
        entries = all_results[key]
        e = [r["energy_w2"] for r in entries]
        q = [r["eq_w2"] for r in entries]
        d = [r["dist_w2"] for r in entries]
        label = "Baseline" if key == "baseline" else f"λ={key.split('_')[1]}"
        print(f"  {label:<18} {np.mean(e):.4f}±{np.std(e):.4f}  {np.mean(q):.4f}±{np.std(q):.4f}  {np.mean(d):.6f}±{np.std(d):.6f}")

    # ── 5. Save JSON ──
    output = {
        "benchmark": "DW4",
        "experiment": "lambda_ablation",
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "seeds": SEEDS,
        "lambdas": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
        "results": all_results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
