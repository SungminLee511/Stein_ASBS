#!/bin/bash
# Launch all 9 DARW ASBS Grid25 experiments (3 betas x 3 seeds)
# Runs 2 at a time (one per GPU), waits for both before launching next pair.
# Logs go into results/<exp_name>/seed_<s>/train.log

cd /home/sky/SML/Stein_ASBS

eval "$(conda shell.bash hook)"
conda activate adjoint_samplers

RESULTS_DIR=results
BETAS=(0.3 0.5 0.7)
SEEDS=(0 1 2)

PAIRS=()
for b in "${BETAS[@]}"; do
  for s in "${SEEDS[@]}"; do
    PAIRS+=("$b $s")
  done
done

i=0
while [ $i -lt ${#PAIRS[@]} ]; do
  read b0 s0 <<< "${PAIRS[$i]}"
  LOG_DIR0="${RESULTS_DIR}/grid25_darw_b${b0}_s${s0}/seed_${s0}"
  mkdir -p "$LOG_DIR0"
  echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching beta=$b0 seed=$s0 on GPU 0"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=grid25_darw_asbs \
    darw_beta=$b0 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=grid25_darw_b${b0}_s${s0} seed=$s0 \
    > "${LOG_DIR0}/train.log" 2>&1 &
  PID0=$!

  PID1=""
  j=$((i + 1))
  if [ $j -lt ${#PAIRS[@]} ]; then
    read b1 s1 <<< "${PAIRS[$j]}"
    LOG_DIR1="${RESULTS_DIR}/grid25_darw_b${b1}_s${s1}/seed_${s1}"
    mkdir -p "$LOG_DIR1"
    echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching beta=$b1 seed=$s1 on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python train.py \
      experiment=grid25_darw_asbs \
      darw_beta=$b1 ksd_lambda=1.0 clip_grad_norm=1.0 \
      exp_name=grid25_darw_b${b1}_s${s1} seed=$s1 \
      > "${LOG_DIR1}/train.log" 2>&1 &
    PID1=$!
  fi

  wait $PID0
  echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Done: beta=$b0 seed=$s0"
  if [ -n "$PID1" ]; then
    wait $PID1
    echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Done: beta=$b1 seed=$s1"
  fi

  i=$((i + 2))
done

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 9 DARW Grid25 experiments finished!"
