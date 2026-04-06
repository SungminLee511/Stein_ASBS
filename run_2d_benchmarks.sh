#!/bin/bash
set -e
cd /home/RESEARCH/Stein_ASBS
PYTHON=/root/miniconda3/envs/Sampling_env/bin/python

echo "=== [$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] Starting 2D benchmark experiments ==="

echo ""
echo "=== [1/6] gmm9_asbs ==="
$PYTHON -u train.py experiment=gmm9_asbs seed=0 use_wandb=false

echo ""
echo "=== [2/6] gmm9_ksd_asbs (lambda=0.01) ==="
$PYTHON -u train.py experiment=gmm9_ksd_asbs seed=0 ksd_lambda=0.01 use_wandb=false

echo ""
echo "=== [3/6] ring8_asbs ==="
$PYTHON -u train.py experiment=ring8_asbs seed=0 use_wandb=false

echo ""
echo "=== [4/6] ring8_ksd_asbs (lambda=0.1) ==="
$PYTHON -u train.py experiment=ring8_ksd_asbs seed=0 ksd_lambda=0.1 use_wandb=false

echo ""
echo "=== [5/6] banana_asbs ==="
$PYTHON -u train.py experiment=banana_asbs seed=0 use_wandb=false

echo ""
echo "=== [6/6] banana_ksd_asbs (lambda=0.5) ==="
$PYTHON -u train.py experiment=banana_ksd_asbs seed=0 use_wandb=false

echo ""
echo "=== [$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')] All 6 experiments done! ==="
