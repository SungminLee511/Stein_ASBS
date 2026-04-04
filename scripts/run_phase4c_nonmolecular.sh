#!/bin/bash
# Phase 4c: Bayesian Logistic Regression — non-molecular benchmark
# Estimated time: ~2 hrs total
set -e

echo "=== Phase 4c: Bayesian Logistic Regression ==="

# Australian dataset (d=15)
echo "--- Australian ---"
for SEED in 0 1 2; do
  echo "BLogReg Australian: Baseline, seed=${SEED}"
  python train.py experiment=blogreg_au_asbs \
    seed=${SEED} use_wandb=false \
    exp_name=blogreg_au_asbs_s${SEED}

  echo "BLogReg Australian: KSD, seed=${SEED}"
  python train.py experiment=blogreg_au_ksd_asbs \
    seed=${SEED} ksd_lambda=1.0 use_wandb=false \
    exp_name=blogreg_au_ksd_s${SEED}
done

# German dataset (d=25)
echo "--- German ---"
for SEED in 0 1 2; do
  echo "BLogReg German: Baseline, seed=${SEED}"
  python train.py experiment=blogreg_ge_asbs \
    seed=${SEED} use_wandb=false \
    exp_name=blogreg_ge_asbs_s${SEED}

  echo "BLogReg German: KSD, seed=${SEED}"
  python train.py experiment=blogreg_ge_ksd_asbs \
    seed=${SEED} ksd_lambda=1.0 use_wandb=false \
    exp_name=blogreg_ge_ksd_s${SEED}
done

echo "=== Phase 4c Complete ==="
