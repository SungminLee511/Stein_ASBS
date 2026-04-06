"""
eval_dw4_bsize_ablation.py — DW4 KSD resample_batch_size ablation evaluation.

Evaluates KSD-ASBS (λ=1.0) at resample_batch_size ∈ {64, 128, 256, 512, 1024}.

Usage:
    python eval_dw4_bsize_ablation.py
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
BSIZE_DIRS = {
    64:   REPO_ROOT / "results" / "dw4_ksd_bsize64"   / "seed_0",
    128:  REPO_ROOT / "results" / "dw4_ksd_bsize128"  / "seed_0",
    256:  REPO_ROOT / "results" / "dw4_ksd_bsize256"  / "seed_0",
    512:  REPO_ROOT / "results" / "dw4_ksd_bsize512"  / "seed_0",
    1024: REPO_ROOT / "results" / "dw4_ksd_bsize1024" / "seed_0",
}

SEEDS = [0, 1, 2, 3, 4]
NUM_EVAL_SAMPLES = 2000
OUTPUT_JSON = EVAL_DIR / "eval_results_dw4_bsize_ablation.json"


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

    # ── Evaluate each batch size ──
    for bsize, run_dir in sorted(BSIZE_DIRS.items()):
        print("\n" + "=" * 70)
        print(f"  Loading KSD-ASBS  resample_batch_size={bsize}")
        print("=" * 70)

        cfg, sde, source, ctrl, evaluator = load_model_from_dir(run_dir, device)
        ctrl.eval()

        key = f"bsize_{bsize}"
        all_results[key] = []
        for seed in SEEDS:
            metrics = generate_and_evaluate(sde, source, cfg, evaluator, seed, NUM_EVAL_SAMPLES, device)
            print(f"  seed={seed}  energy_W2={metrics['energy_w2']:.4f}  eq_W2={metrics['eq_w2']:.4f}  dist_W2={metrics['dist_w2']:.6f}")
            all_results[key].append({"seed": seed, **metrics})

        del sde, source, ctrl, evaluator
        torch.cuda.empty_cache()

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  SUMMARY — DW4 resample_batch_size Ablation (KSD λ=1.0)")
    print("=" * 70)
    print(f"\n  {'Batch Size':<18} {'energy_W2':>14} {'eq_W2':>14} {'dist_W2':>14}")
    print(f"  {'─' * 18} {'─' * 14} {'─' * 14} {'─' * 14}")

    for bsize in [64, 128, 256, 512, 1024]:
        key = f"bsize_{bsize}"
        if key not in all_results:
            continue
        entries = all_results[key]
        e = [r["energy_w2"] for r in entries]
        q = [r["eq_w2"] for r in entries]
        d = [r["dist_w2"] for r in entries]
        print(f"  B={bsize:<14} {np.mean(e):.4f}±{np.std(e):.4f}  {np.mean(q):.4f}±{np.std(q):.4f}  {np.mean(d):.6f}±{np.std(d):.6f}")

    # ── Save JSON ──
    output = {
        "benchmark": "DW4",
        "experiment": "resample_batch_size_ablation",
        "ksd_lambda": 1.0,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "seeds": SEEDS,
        "batch_sizes": [64, 128, 256, 512, 1024],
        "results": all_results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
