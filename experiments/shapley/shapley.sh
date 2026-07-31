#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=shapley-%j.out
#SBATCH --error=shapley-%j.err

DATASET=$1
DATA=$2
EXP=$3
CKPT=$4
BATCHSIZE=128
REPLICATE=($(seq 1 1 10))

echo $DATASET $DATA $EXP $CKPT

source $HOME/ssetup.sh $DATASET


parallel --delay=2 --jobs=1  \
    python -m brute_force_shapley  \
        --ckpt="$EXP/replicate_{1}/level_0/main/model_$CKPT.pth"  \
        --path_to_data_hparam="$DATA"  \
        --train  \
        --val_samples='1,100,10000'  \
        --verbose  \
        --batch_size=$BATCHSIZE  \
    ::: ${REPLICATE[@]}  \
