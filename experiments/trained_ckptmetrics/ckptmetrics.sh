#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=gen-metrics-%j.out
#SBATCH --error=gen-metrics-%j.err

DATASET=$1
DATA=$2
EXP=$3
CKPT=$4
BATCHSIZE=50
REPLICATE=($(seq 1 1 100))

echo $DATASET $DATA $EXP $CKPT

source $HOME/ssetup.sh $DATASET
pip install faiss-gpu==1.7.2

python -m generate_ensemble_metrics  \
    --exp="$EXP"  \
    --path_to_data_hparam="$DATA"  \
    --train  \
    --verbose  \
    --batch_size=$BATCHSIZE  \
    --ep_it="$CKPT"  \

parallel --delay=5 --jobs=1  \
    python -m generate_single_ckptmetric  \
        --ckpt="$EXP/replicate_{1}/level_0/main/model_$CKPT.pth"  \
        --path_to_data_hparam="$DATA"  \
        --train  \
        --verbose  \
        --batch_size=$BATCHSIZE  \
    ::: ${REPLICATE[@]}  \
