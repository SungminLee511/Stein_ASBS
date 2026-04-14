#!/bin/bash
# Run a single MW5 SDR ASBS experiment with NaN kill switch.
# Usage: bash run_mw5_sdr_with_nan_guard.sh <gpu_id> <beta> <seed> <exp_name> <logfile>

GPU_ID=$1
BETA=$2
SEED=$3
EXP_NAME=$4
LOGFILE=$5

cd /home/sky/SML/Stein_ASBS

# Launch training in background
PYTHONPATH=/home/sky/SML/Stein_ASBS \
CUDA_VISIBLE_DEVICES=$GPU_ID \
/home/sky/miniconda3/envs/adjoint_samplers/bin/python train.py \
  experiment=mw5_sdr_asbs \
  sdr_beta=$BETA \
  sdr_lambda=1.0 \
  clip_grad_norm=1.0 \
  exp_name=$EXP_NAME \
  seed=$SEED \
  > "$LOGFILE" 2>&1 &

TRAIN_PID=$!
echo "[NaN guard] Started $EXP_NAME (PID=$TRAIN_PID, GPU=$GPU_ID, beta=$BETA, seed=$SEED)"

# Monitor loop: check last 5 lines of log for NaN every 10 seconds
while kill -0 $TRAIN_PID 2>/dev/null; do
  sleep 10
  if tail -5 "$LOGFILE" 2>/dev/null | grep -q "loss=nan"; then
    # Extract epoch from the NaN line
    NAN_EPOCH=$(tail -5 "$LOGFILE" | grep "loss=nan" | head -1 | grep -oP 'ep=\K[0-9]+')
    echo "[NaN guard] *** Loss became NaN at epoch $NAN_EPOCH — $EXP_NAME terminated ***"
    kill $TRAIN_PID 2>/dev/null
    wait $TRAIN_PID 2>/dev/null
    exit 1
  fi
done

# Training finished normally
wait $TRAIN_PID
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "[NaN guard] $EXP_NAME completed successfully."
else
  echo "[NaN guard] $EXP_NAME exited with code $EXIT_CODE."
fi
exit $EXIT_CODE
