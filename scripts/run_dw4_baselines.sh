#!/bin/bash
# DW4 baseline experiments: AS, iDEM, pDEM, DGFS
# No evaluations during training, checkpoints every 100 iterations
# All run in parallel on same GPU

PYTHON=/root/miniconda3/envs/Sampling_env/bin/python
PROJECT_DIR=/home/RESEARCH/Stein_ASBS
RESULTS_DIR=$PROJECT_DIR/results

echo "=== Launching DW4 baseline experiments ==="
echo "$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M:%S KST')"

# 1. AS (Adjoint Sampling) — 5000 epochs, save every 100, skip eval
echo "[1/4] Launching dw4_as..."
cd $PROJECT_DIR
nohup $PYTHON -u train.py \
    experiment=dw4_as \
    skip_eval=true \
    save_freq=100 \
    seed=0 \
    use_wandb=false \
    > $RESULTS_DIR/dw4_as_train.log 2>&1 &
echo "  PID: $!"

# 2. iDEM — 1000 epochs, no validation, checkpoint every 10 epochs
echo "[2/4] Launching dw4_idem..."
cd $PROJECT_DIR/baseline_models/dem
nohup $PYTHON -u dem/train.py \
    experiment=dw4_idem \
    logger=csv \
    trainer.max_epochs=1000 \
    trainer.check_val_every_n_epoch=99999 \
    callbacks=default \
    callbacks.model_checkpoint.monitor=null \
    callbacks.model_checkpoint.save_top_k=-1 \
    callbacks.model_checkpoint.every_n_epochs=10 \
    callbacks.model_checkpoint.save_on_train_epoch_end=true \
    test=false \
    hydra.run.dir=$RESULTS_DIR/dw4_idem \
    > $RESULTS_DIR/dw4_idem_train.log 2>&1 &
echo "  PID: $!"

# 3. pDEM — 1000 epochs, no validation, checkpoint every 10 epochs
echo "[3/4] Launching dw4_pdem..."
cd $PROJECT_DIR/baseline_models/dem
nohup $PYTHON -u dem/train.py \
    experiment=dw4_pdem \
    logger=csv \
    trainer.max_epochs=1000 \
    trainer.check_val_every_n_epoch=99999 \
    callbacks=default \
    callbacks.model_checkpoint.monitor=null \
    callbacks.model_checkpoint.save_top_k=-1 \
    callbacks.model_checkpoint.every_n_epochs=10 \
    callbacks.model_checkpoint.save_on_train_epoch_end=true \
    test=false \
    hydra.run.dir=$RESULTS_DIR/dw4_pdem \
    > $RESULTS_DIR/dw4_pdem_train.log 2>&1 &
echo "  PID: $!"

# 4. DGFS — 5000 steps, skip eval, save every 100 steps
echo "[4/4] Launching dw4_dgfs..."
cd $PROJECT_DIR/baseline_models/dgfs
nohup $PYTHON -u gflownet/main.py \
    target=dw4 \
    steps=5000 \
    skip_eval=true \
    save_freq=100 \
    f_func.in_shape=8 \
    f_func.out_shape=8 \
    hydra.run.dir=$RESULTS_DIR/dw4_dgfs \
    > $RESULTS_DIR/dw4_dgfs_train.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== All 4 experiments launched ==="
echo "Log files:"
echo "  AS:   $RESULTS_DIR/dw4_as_train.log"
echo "  iDEM: $RESULTS_DIR/dw4_idem_train.log"
echo "  pDEM: $RESULTS_DIR/dw4_pdem_train.log"
echo "  DGFS: $RESULTS_DIR/dw4_dgfs_train.log"
