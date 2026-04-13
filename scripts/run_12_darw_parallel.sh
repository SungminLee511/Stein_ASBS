#!/bin/bash
# Launch 12 DARW experiments: 6 per GPU, all in parallel
# GPU 0: grid25 β=1.0 (3 seeds) + mw5 β=0.5 (3 seeds)
# GPU 1: mw5 β=0.7 (3 seeds) + mw5 β=1.0 (3 seeds)

cd /home/sky/SML/Stein_ASBS

eval "$(conda shell.bash hook)"
conda activate adjoint_samplers

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching 12 DARW experiments..."

# === GPU 0: 6 jobs ===

# Grid25 DARW β=1.0, seeds 0,1,2
for s in 0 1 2; do
  echo "  GPU0: grid25 darw b1.0 seed=$s"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=grid25_darw_asbs \
    darw_beta=1.0 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=grid25_darw_b1.0_s${s} seed=$s \
    > log_grid25_darw_b1.0_s${s}.txt 2>&1 &
done

# MW5 DARW β=0.5, seeds 0,1,2
for s in 0 1 2; do
  echo "  GPU0: mw5 darw b0.5 seed=$s"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=mw5_darw_asbs \
    darw_beta=0.5 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_darw_b0.5_s${s} seed=$s \
    > log_mw5_darw_b0.5_s${s}.txt 2>&1 &
done

# === GPU 1: 6 jobs ===

# MW5 DARW β=0.7, seeds 0,1,2
for s in 0 1 2; do
  echo "  GPU1: mw5 darw b0.7 seed=$s"
  CUDA_VISIBLE_DEVICES=1 python train.py \
    experiment=mw5_darw_asbs \
    darw_beta=0.7 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_darw_b0.7_s${s} seed=$s \
    > log_mw5_darw_b0.7_s${s}.txt 2>&1 &
done

# MW5 DARW β=1.0, seeds 0,1,2
for s in 0 1 2; do
  echo "  GPU1: mw5 darw b1.0 seed=$s"
  CUDA_VISIBLE_DEVICES=1 python train.py \
    experiment=mw5_darw_asbs \
    darw_beta=1.0 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=mw5_darw_b1.0_s${s} seed=$s \
    > log_mw5_darw_b1.0_s${s}.txt 2>&1 &
done

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 12 jobs launched. Waiting..."
wait
echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 12 DARW experiments finished!"
