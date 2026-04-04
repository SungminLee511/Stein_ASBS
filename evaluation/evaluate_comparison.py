"""
evaluate_comparison.py — Head-to-head evaluation of baseline ASBS vs KSD-ASBS.

Loads checkpoints, generates samples with multiple seeds, evaluates W2 metrics,
and prints a comparison table. Outputs a JSON file for generate_results.py.

Usage:
    python evaluate_comparison.py \
        --baseline_dir baselines/dw4_asbs \
        --ksd_dir results/local/2026.04.04/064017 \
        --seeds 0 1 2 3 4 \
        --num_eval_samples 2000 \
        --output_json eval_results_dw4.json
"""

import argparse
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


def load_model_from_dir(run_dir: str, device: str = "cuda"):
    """Load controller + corrector + SDE from a run directory with config.yaml + checkpoints/."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "checkpoint_latest.pt"

    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    cfg = OmegaConf.load(cfg_path)

    # Instantiate energy, source, SDE, controller
    energy = hydra.utils.instantiate(cfg.energy, device=device)
    source = hydra.utils.instantiate(cfg.source, device=device)
    ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
    controller = hydra.utils.instantiate(cfg.controller).to(device)
    sde = ControlledSDE(ref_sde, controller).to(device)

    # Load checkpoint weights (controller only — we just need to generate samples)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(checkpoint["controller"])

    # Instantiate evaluator
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
    parser = argparse.ArgumentParser(description="Evaluate baseline vs KSD-ASBS")
    parser.add_argument("--baseline_dir", type=str, required=True, help="Path to baseline run dir")
    parser.add_argument("--ksd_dir", type=str, required=True, help="Path to KSD-ASBS run dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_eval_samples", type=int, default=2000)
    parser.add_argument("--output_json", type=str, default="eval_results_dw4.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load models ──
    print("\n" + "=" * 60)
    print("Loading BASELINE model...")
    print("=" * 60)
    bl_cfg, bl_sde, bl_source, bl_ctrl, bl_eval = load_model_from_dir(args.baseline_dir, device)
    bl_ctrl.eval()

    print("\n" + "=" * 60)
    print("Loading KSD-ASBS model...")
    print("=" * 60)
    ksd_cfg, ksd_sde, ksd_source, ksd_ctrl, ksd_eval = load_model_from_dir(args.ksd_dir, device)
    ksd_ctrl.eval()

    # ── Evaluate over seeds ──
    results = {"baseline": [], "ksd_asbs": []}

    for seed in args.seeds:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed}")
        print(f"{'─' * 60}")

        print(f"  [Baseline] Generating {args.num_eval_samples} samples...")
        bl_metrics = generate_and_evaluate(bl_sde, bl_source, bl_cfg, bl_eval, seed, args.num_eval_samples, device)
        print(f"  [Baseline] energy_W2={bl_metrics['energy_w2']:.4f}  eq_W2={bl_metrics['eq_w2']:.4f}  dist_W2={bl_metrics['dist_w2']:.6f}")
        results["baseline"].append({"seed": seed, **bl_metrics})

        print(f"  [KSD-ASBS] Generating {args.num_eval_samples} samples...")
        ksd_metrics = generate_and_evaluate(ksd_sde, ksd_source, ksd_cfg, ksd_eval, seed, args.num_eval_samples, device)
        print(f"  [KSD-ASBS] energy_W2={ksd_metrics['energy_w2']:.4f}  eq_W2={ksd_metrics['eq_w2']:.4f}  dist_W2={ksd_metrics['dist_w2']:.6f}")
        results["ksd_asbs"].append({"seed": seed, **ksd_metrics})

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY — DW4 Evaluation")
    print("=" * 60)

    for method_name, method_results in results.items():
        energies = [r["energy_w2"] for r in method_results]
        eqs = [r["eq_w2"] for r in method_results]
        dists = [r["dist_w2"] for r in method_results]

        print(f"\n  {method_name.upper()}")
        print(f"    energy_W2:  {np.mean(energies):.4f} ± {np.std(energies):.4f}  (best: {np.min(energies):.4f}, seed={method_results[np.argmin(energies).item()]['seed']})")
        print(f"    eq_W2:      {np.mean(eqs):.4f} ± {np.std(eqs):.4f}  (best: {np.min(eqs):.4f}, seed={method_results[np.argmin(eqs).item()]['seed']})")
        print(f"    dist_W2:    {np.mean(dists):.6f} ± {np.std(dists):.6f}  (best: {np.min(dists):.6f}, seed={method_results[np.argmin(dists).item()]['seed']})")

    # ── Improvement ──
    bl_energy_mean = np.mean([r["energy_w2"] for r in results["baseline"]])
    ksd_energy_mean = np.mean([r["energy_w2"] for r in results["ksd_asbs"]])
    bl_eq_mean = np.mean([r["eq_w2"] for r in results["baseline"]])
    ksd_eq_mean = np.mean([r["eq_w2"] for r in results["ksd_asbs"]])
    bl_dist_mean = np.mean([r["dist_w2"] for r in results["baseline"]])
    ksd_dist_mean = np.mean([r["dist_w2"] for r in results["ksd_asbs"]])

    print(f"\n  IMPROVEMENT (KSD-ASBS vs Baseline)")
    for name, bl_val, ksd_val in [
        ("energy_W2", bl_energy_mean, ksd_energy_mean),
        ("eq_W2", bl_eq_mean, ksd_eq_mean),
        ("dist_W2", bl_dist_mean, ksd_dist_mean),
    ]:
        if bl_val > 0:
            pct = (bl_val - ksd_val) / bl_val * 100
            direction = "↓ better" if pct > 0 else "↑ worse"
            print(f"    {name}: {pct:+.2f}% ({direction})")

    # ── Save JSON ──
    output = {
        "benchmark": "DW4",
        "num_eval_samples": args.num_eval_samples,
        "seeds": args.seeds,
        "baseline_dir": str(args.baseline_dir),
        "ksd_dir": str(args.ksd_dir),
        "results": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
