#!/bin/bash
# LJ13 experiments: 4 configs × 3 seeds = 12 total
# Parallel per seed, sequential across seeds
cd /home/RESEARCH/Stein_ASBS
PY=/root/miniconda3/envs/asbs_aldp/bin/python

echo "=== LJ13 Full Experiment Suite ==="
echo "Start: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"

for SEED in 0 1 2; do
    echo ""
    echo "=========================================="
    echo "  SEED ${SEED} — launching 4 jobs in parallel"
    echo "  $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
    echo "=========================================="

    # 1) Vanilla ASBS
    $PY -u train.py experiment=lj13_asbs lancher=local skip_eval=true use_wandb=false seed=${SEED} \
        exp_name=lj13_asbs_s${SEED} \
        > /home/RESEARCH/Stein_ASBS/lj13_asbs_s${SEED}.log 2>&1 &
    PID1=$!
    echo "[PID $PID1] lj13_asbs seed=${SEED}"

    # 2) SDR ASBS β=0.5
    $PY -u train.py experiment=lj13_sdr_asbs lancher=local skip_eval=true use_wandb=false seed=${SEED} \
        sdr_beta=0.5 exp_name=lj13_sdr_b0.5_s${SEED} \
        > /home/RESEARCH/Stein_ASBS/lj13_sdr_b0.5_s${SEED}.log 2>&1 &
    PID2=$!
    echo "[PID $PID2] lj13_sdr_asbs β=0.5 seed=${SEED}"

    # 3) SDR ASBS β=0.7
    $PY -u train.py experiment=lj13_sdr_asbs lancher=local skip_eval=true use_wandb=false seed=${SEED} \
        sdr_beta=0.7 exp_name=lj13_sdr_b0.7_s${SEED} \
        > /home/RESEARCH/Stein_ASBS/lj13_sdr_b0.7_s${SEED}.log 2>&1 &
    PID3=$!
    echo "[PID $PID3] lj13_sdr_asbs β=0.7 seed=${SEED}"

    # 4) SDR ASBS β=1.0
    $PY -u train.py experiment=lj13_sdr_asbs lancher=local skip_eval=true use_wandb=false seed=${SEED} \
        sdr_beta=1.0 exp_name=lj13_sdr_b1.0_s${SEED} \
        > /home/RESEARCH/Stein_ASBS/lj13_sdr_b1.0_s${SEED}.log 2>&1 &
    PID4=$!
    echo "[PID $PID4] lj13_sdr_asbs β=1.0 seed=${SEED}"

    echo "Waiting for seed ${SEED} jobs to finish..."
    wait $PID1 $PID2 $PID3 $PID4
    echo "Seed ${SEED} DONE at $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
done

echo ""
echo "=== ALL 12 EXPERIMENTS COMPLETE ==="
echo "End: $(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"
