#!/bin/bash
# Phase 4: Rotated GMM experiments — demonstrating advantage when CVs are unknown
set -e

echo "=== Phase 4: Synthetic CV-Unknown Experiments ==="

# For each dimension: train baseline ASBS and KSD-ASBS
for DIM in 10 30 50 100; do
  for SEED in 0 1 2; do
    echo "RotGMM d=${DIM}: Baseline, seed=${SEED}"
    python train.py experiment=rotgmm${DIM}_asbs \
      seed=${SEED} use_wandb=false \
      exp_name=rotgmm${DIM}_asbs_s${SEED}

    echo "RotGMM d=${DIM}: KSD, seed=${SEED}"
    python train.py experiment=rotgmm${DIM}_ksd_asbs \
      seed=${SEED} ksd_lambda=1.0 use_wandb=false \
      exp_name=rotgmm${DIM}_ksd_s${SEED}
  done
done

echo "=== Phase 4 Complete ==="
