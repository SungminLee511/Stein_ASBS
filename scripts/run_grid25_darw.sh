#!/bin/bash
# Launch all 9 DARW ASBS Grid25 experiments (3 betas x 3 seeds)
# Runs 2 at a time (one per GPU), waits for both before launching next pair.

cd /home/sky/SML/Stein_ASBS

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adjoint_samplers

BETAS=(0.3 0.5 0.7)
SEEDS=(0 1 2)

# Build list of all (beta, seed) pairs
PAIRS=()
for b in "${BETAS[@]}"; do
  for s in "${SEEDS[@]}"; do
    PAIRS+=("$b $s")
  done
done

# Run 2 at a time
i=0
while [ $i -lt ${#PAIRS[@]} ]; do
  # Launch on GPU 0
  read b0 s0 <<< "${PAIRS[$i]}"
  echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching beta=$b0 seed=$s0 on GPU 0"
  CUDA_VISIBLE_DEVICES=0 python train.py \
    experiment=grid25_darw_asbs \
    darw_beta=$b0 ksd_lambda=1.0 clip_grad_norm=1.0 \
    exp_name=grid25_darw_b${b0}_s${s0} seed=$s0 \
    > log_grid25_darw_b${b0}_s${s0}.txt 2>&1 &
  PID0=$!

  # Launch on GPU 1 if there's another experiment
  PID1=""
  j=$((i + 1))
  if [ $j -lt ${#PAIRS[@]} ]; then
    read b1 s1 <<< "${PAIRS[$j]}"
    echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Launching beta=$b1 seed=$s1 on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python train.py \
      experiment=grid25_darw_asbs \
      darw_beta=$b1 ksd_lambda=1.0 clip_grad_norm=1.0 \
      exp_name=grid25_darw_b${b1}_s${s1} seed=$s1 \
      > log_grid25_darw_b${b1}_s${s1}.txt 2>&1 &
    PID1=$!
  fi

  # Wait for both
  wait $PID0
  echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Done: beta=$b0 seed=$s0"
  if [ -n "$PID1" ]; then
    wait $PID1
    echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Done: beta=$b1 seed=$s1"
  fi

  i=$((i + 2))
done

echo "[$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 9 DARW Grid25 experiments finished!"
