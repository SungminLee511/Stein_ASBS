# Copyright (c) Meta Platforms, Inc. and affiliates.

# ASBS (baseline)
python train.py \
    experiment=lj38_asbs \
    adj_num_epochs_per_stage=200,300 \
    seed=0,1,2 \
    -m &

# IMQ-KSD-ASBS
python train.py \
    experiment=lj38_imq_asbs \
    adj_num_epochs_per_stage=200,300 \
    seed=0,1,2 \
    -m &
