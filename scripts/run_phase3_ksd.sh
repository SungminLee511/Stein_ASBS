#!/bin/bash
# Phase 3: KSD-augmented ASBS training with lambda ablation
set -e

echo "=== Phase 3: KSD-ASBS Training ==="

# --- DW4 lambda ablation (5 lambda x 3 seeds = 15 runs) ---
echo "--- DW4 KSD Lambda Ablation ---"
for LAMBDA in 0.1 0.5 1.0 5.0 10.0; do
  for SEED in 0 1 2; do
    echo "DW4: lambda=${LAMBDA}, seed=${SEED}"
    python train.py experiment=dw4_ksd_asbs \
      seed=${SEED} \
      ksd_lambda=${LAMBDA} \
      use_wandb=false \
      exp_name=dw4_ksd_l${LAMBDA}_s${SEED}
  done
done

# --- LJ13 (3 lambda x 3 seeds = 9 runs) ---
echo "--- LJ13 KSD ---"
for LAMBDA in 0.5 1.0 5.0; do
  for SEED in 0 1 2; do
    echo "LJ13: lambda=${LAMBDA}, seed=${SEED}"
    python train.py experiment=lj13_ksd_asbs \
      seed=${SEED} \
      ksd_lambda=${LAMBDA} \
      use_wandb=false \
      exp_name=lj13_ksd_l${LAMBDA}_s${SEED}
  done
done

# --- LJ38 (3 lambda x 3 seeds = 9 runs) ---
echo "--- LJ38 KSD ---"
for LAMBDA in 0.5 1.0 5.0; do
  for SEED in 0 1 2; do
    echo "LJ38: lambda=${LAMBDA}, seed=${SEED}"
    python train.py experiment=lj38_ksd_asbs \
      seed=${SEED} \
      ksd_lambda=${LAMBDA} \
      use_wandb=false \
      exp_name=lj38_ksd_l${LAMBDA}_s${SEED}
  done
done

# --- LJ55 (best lambda only, 3 seeds) ---
echo "--- LJ55 KSD ---"
BEST_LAMBDA=1.0  # Update after DW4/LJ13 ablation
for SEED in 0 1 2; do
  echo "LJ55: lambda=${BEST_LAMBDA}, seed=${SEED}"
  python train.py experiment=lj55_ksd_asbs \
    seed=${SEED} \
    ksd_lambda=${BEST_LAMBDA} \
    use_wandb=false \
    exp_name=lj55_ksd_l${BEST_LAMBDA}_s${SEED}
done

# --- Muller-Brown KSD (3 seeds) ---
echo "--- Muller-Brown KSD ---"
for SEED in 0 1 2; do
  echo "Muller-Brown KSD: seed=${SEED}"
  python train.py experiment=muller_ksd_asbs seed=${SEED} ksd_lambda=1.0 \
    use_wandb=false exp_name=muller_ksd_s${SEED}
done

# --- Bayesian LogReg KSD (2 datasets x 3 seeds) ---
echo "--- Bayesian LogReg KSD ---"
for DATASET in au ge; do
  for SEED in 0 1 2; do
    echo "BLogReg ${DATASET} KSD: seed=${SEED}"
    python train.py experiment=blogreg_${DATASET}_ksd_asbs seed=${SEED} ksd_lambda=1.0 \
      use_wandb=false exp_name=blogreg_${DATASET}_ksd_s${SEED}
  done
done

echo "=== Phase 3 Complete ==="
