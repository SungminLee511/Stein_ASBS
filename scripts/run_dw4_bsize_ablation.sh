#!/bin/bash
# DW4 Batch Size Ablation: resample_batch_size in {64, 128, 256, 512, 1024}
# Fixed: ksd_lambda=1.0, seed=0
set -e

cd /home/RESEARCH/Stein_ASBS

PYTHON=/root/miniconda3/envs/Sampling_env/bin/python

for BSIZE in 64 128 256 512 1024; do
  echo "=========================================="
  echo "Starting DW4 KSD-ASBS batch_size=${BSIZE}"
  echo "Time: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo "=========================================="

  OUTDIR="results/dw4_ksd_bsize${BSIZE}/seed_0"
  mkdir -p "${OUTDIR}"

  $PYTHON -u train.py experiment=dw4_ksd_asbs seed=0 ksd_lambda=1.0 \
    resample_batch_size=${BSIZE} \
    use_wandb=false \
    exp_name=dw4_ksd_bsize${BSIZE} \
    save_freq=50

  echo "Finished batch_size=${BSIZE} at $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
  echo ""
done

echo "=== All DW4 batch size ablation experiments complete ==="
echo "Time: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
