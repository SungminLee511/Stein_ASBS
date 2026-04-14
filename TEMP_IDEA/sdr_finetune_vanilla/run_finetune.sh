#!/bin/bash
# ======================================================================
# SDR → Vanilla ASBS Fine-tuning
#
# Loads a trained SDR-ASBS checkpoint and continues training with
# sdr_lambda=0 (no KSD correction) to fix Energy W2 while retaining
# mode coverage from SDR.
#
# Usage:
#   bash TEMP_IDEA/sdr_finetune_vanilla/run_finetune.sh <benchmark> <sdr_result_dir> <seed> [gpu]
#
# Examples:
#   bash TEMP_IDEA/sdr_finetune_vanilla/run_finetune.sh grid25 results/grid25_sdr_b0.7_s0 0 0
#   bash TEMP_IDEA/sdr_finetune_vanilla/run_finetune.sh mw5 results/mw5_sdr_b0.7_s0 0 1
# ======================================================================

set -euo pipefail

BENCHMARK="${1:?Usage: $0 <grid25|mw5> <sdr_result_dir> <seed> [gpu]}"
SDR_DIR="${2:?Provide SDR result directory (e.g. results/grid25_sdr_b0.7_s0)}"
SEED="${3:?Provide seed}"
GPU="${4:-0}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/sky/miniconda3/envs/adjoint_samplers/bin/python}"
IDEA_DIR="$REPO/TEMP_IDEA/sdr_finetune_vanilla"

# --- Resolve checkpoint ---
CKPT="$REPO/$SDR_DIR/seed_${SEED}/results/$(basename "$SDR_DIR")/seed_${SEED}/checkpoints/checkpoint_latest.pt"
if [ ! -f "$CKPT" ]; then
    # Try flat structure
    CKPT="$REPO/$SDR_DIR/checkpoints/checkpoint_latest.pt"
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Cannot find checkpoint in $SDR_DIR"
    echo "Tried: $REPO/$SDR_DIR/seed_${SEED}/results/*/seed_${SEED}/checkpoints/checkpoint_latest.pt"
    exit 1
fi
echo "Checkpoint: $CKPT"

# --- Output directory ---
SDR_BASE="$(basename "$SDR_DIR")"
OUT_NAME="${SDR_BASE}_finetune_s${SEED}"
OUT_DIR="$IDEA_DIR/results/$OUT_NAME"
mkdir -p "$OUT_DIR"

# --- Config ---
CONFIG_NAME="${BENCHMARK}_finetune"

# --- Run ---
echo "=== Fine-tuning $BENCHMARK | SDR=$SDR_DIR | seed=$SEED | GPU=$GPU ==="
echo "Output: $OUT_DIR"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" $PY train.py \
    --config-path="$IDEA_DIR/configs" \
    --config-name="$CONFIG_NAME" \
    exp_name="$OUT_NAME" \
    seed="$SEED" \
    checkpoint="$CKPT" \
    hydra.run.dir="$OUT_DIR" \
    "$@"
