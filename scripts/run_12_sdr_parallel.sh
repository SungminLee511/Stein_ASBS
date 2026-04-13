#!/bin/bash
# Launch 12 SDR experiments: 6 per GPU, all in parallel
# GPU 0: grid25 β=1.0 (3 seeds) + mw5 β=0.5 (3 seeds)
# GPU 1: mw5 β=0.7 (3 seeds) + mw5 β=1.0 (3 seeds)
# Logs go into results/<exp_name>/seed_<s>/train.log

cd /home/sky/SML/Stein_ASBS

eval "$(conda shell.bash hook)"
conda activate adjoint_samplers

RESULTS_DIR=results

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching 12 SDR experiments..."

# === GPU 0: 6 jobs ===

# Grid25 SDR β=1.0, seeds 0,1,2
for s in 0 1 2; do
  LOG_DIR="${RESULTS_DIR}/grid25_sdr_b1.0_s${s}/seed_${s}"
  mkdir -p "$LOG_DIR"
  echo "  GPU0: grid25 sdr b1.0 seed=$s -> $LOG_DIR/train.log"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=grid25_sdr_asbs \
    sdr_beta=1.0 sdr_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=grid25_sdr_b1.0_s${s} seed=$s \
    > "${LOG_DIR}/train.log" 2>&1 &
done

# MW5 SDR β=0.5, seeds 0,1,2
for s in 0 1 2; do
  LOG_DIR="${RESULTS_DIR}/mw5_sdr_b0.5_s${s}/seed_${s}"
  mkdir -p "$LOG_DIR"
  echo "  GPU0: mw5 sdr b0.5 seed=$s -> $LOG_DIR/train.log"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=mw5_sdr_asbs \
    sdr_beta=0.5 sdr_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_sdr_b0.5_s${s} seed=$s \
    > "${LOG_DIR}/train.log" 2>&1 &
done

# === GPU 1: 6 jobs ===

# MW5 SDR β=0.7, seeds 0,1,2
for s in 0 1 2; do
  LOG_DIR="${RESULTS_DIR}/mw5_sdr_b0.7_s${s}/seed_${s}"
  mkdir -p "$LOG_DIR"
  echo "  GPU1: mw5 sdr b0.7 seed=$s -> $LOG_DIR/train.log"
  CUDA_VISIBLE_DEVICES=1 python train.py \
    experiment=mw5_sdr_asbs \
    sdr_beta=0.7 sdr_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_sdr_b0.7_s${s} seed=$s \
    > "${LOG_DIR}/train.log" 2>&1 &
done

# MW5 SDR β=1.0, seeds 0,1,2
for s in 0 1 2; do
  LOG_DIR="${RESULTS_DIR}/mw5_sdr_b1.0_s${s}/seed_${s}"
  mkdir -p "$LOG_DIR"
  echo "  GPU1: mw5 sdr b1.0 seed=$s -> $LOG_DIR/train.log"
  CUDA_VISIBLE_DEVICES=1 python train.py \
    experiment=mw5_sdr_asbs \
    sdr_beta=1.0 sdr_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_sdr_b1.0_s${s} seed=$s \
    > "${LOG_DIR}/train.log" 2>&1 &
done

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 12 jobs launched. Waiting..."
wait
echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 12 SDR experiments finished!"
