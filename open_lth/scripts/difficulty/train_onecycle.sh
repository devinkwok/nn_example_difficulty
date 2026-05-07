#!/bin/bash

set -x

MODEL=$1
DATASET=$2
REPLICATE=$3

SEED=$RANDOM

REPLICATE_OFFSET=$SLURM_PROCID
REPLICATE=$(($REPLICATE + $REPLICATE_OFFSET))

python open_lth.py lottery  \
    --default_hparams=$MODEL  \
    --dataset_name=$DATASET  \
    --replicate=$REPLICATE  \
    --data_order_seed=$SEED  \
    --levels=0  \
    --training_steps=160ep  \
    --save_ckpt_steps="0ep,10ep,20ep,150ep,160ep"  \
    --metrics_n_train=50000  \
    --pointwise_metrics_steps="1ep-160ep"  \
    --grad_metrics_steps="10ep,20ep"  \
    --grad_metrics_batch_size=20  \
    --batch_forget_track  \
        --lr=0.1  \
        --lr_schedule=onecycle  \
        --warmup_steps="1ep"  \
