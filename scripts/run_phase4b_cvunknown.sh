#!/bin/bash
# Phase 4b: Muller-Brown experiments — 2D visualization benchmark
# Estimated time: ~30 min total
set -e

echo "=== Phase 4b: Muller-Brown (CV-Unknown Visualization) ==="

# Baseline
for SEED in 0 1 2; do
  echo "Muller-Brown: Baseline, seed=${SEED}"
  python train.py experiment=muller_asbs \
    seed=${SEED} use_wandb=false \
    exp_name=muller_asbs_s${SEED}
done

# KSD
for SEED in 0 1 2; do
  echo "Muller-Brown: KSD, seed=${SEED}"
  python train.py experiment=muller_ksd_asbs \
    seed=${SEED} ksd_lambda=1.0 use_wandb=false \
    exp_name=muller_ksd_s${SEED}
done

echo "=== Phase 4b Complete ==="
