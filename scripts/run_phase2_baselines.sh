#!/bin/bash
# Phase 2: Train all baselines
# Estimated time: ~50 GPU-hours total
set -e

echo "=== Phase 2: Baseline ASBS Training ==="

# Download reference samples
bash scripts/download.sh

# DW4 baselines (3 seeds, ~1 hr each)
echo "--- DW4 Baselines ---"
for SEED in 0 1 2; do
  echo "DW4: seed=${SEED}"
  python train.py experiment=dw4_asbs seed=${SEED} use_wandb=false exp_name=dw4_asbs_s${SEED}
done

# LJ13 baselines (3 seeds, ~4 hrs each)
echo "--- LJ13 Baselines ---"
for SEED in 0 1 2; do
  echo "LJ13: seed=${SEED}"
  python train.py experiment=lj13_asbs seed=${SEED} use_wandb=false exp_name=lj13_asbs_s${SEED}
done

# LJ38 baselines (3 seeds, ~8 hrs each)
echo "--- LJ38 Baselines ---"
for SEED in 0 1 2; do
  echo "LJ38: seed=${SEED}"
  python train.py experiment=lj38_asbs seed=${SEED} use_wandb=false exp_name=lj38_asbs_s${SEED}
done

# LJ55 baselines (3 seeds, ~12 hrs each)
echo "--- LJ55 Baselines ---"
for SEED in 0 1 2; do
  echo "LJ55: seed=${SEED}"
  python train.py experiment=lj55_asbs seed=${SEED} use_wandb=false exp_name=lj55_asbs_s${SEED}
done

# Muller-Brown baselines (3 seeds, ~10 min each)
echo "--- Muller-Brown Baselines ---"
for SEED in 0 1 2; do
  echo "Muller-Brown: seed=${SEED}"
  python train.py experiment=muller_asbs seed=${SEED} use_wandb=false exp_name=muller_asbs_s${SEED}
done

# Bayesian LogReg baselines (2 datasets x 3 seeds, ~20 min each)
echo "--- Bayesian LogReg Baselines ---"
for DATASET in au ge; do
  for SEED in 0 1 2; do
    echo "BLogReg ${DATASET}: seed=${SEED}"
    python train.py experiment=blogreg_${DATASET}_asbs seed=${SEED} use_wandb=false \
      exp_name=blogreg_${DATASET}_asbs_s${SEED}
  done
done

# RotGMM baselines (4 dims x 3 seeds, ~20 min each)
echo "--- RotGMM Baselines ---"
for DIM in 10 30 50 100; do
  for SEED in 0 1 2; do
    echo "RotGMM d=${DIM}: seed=${SEED}"
    python train.py experiment=rotgmm${DIM}_asbs seed=${SEED} use_wandb=false \
      exp_name=rotgmm${DIM}_asbs_s${SEED}
  done
done

echo "=== Phase 2 Complete ==="
