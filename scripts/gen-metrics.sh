#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

# to run on Mila cluster
source ./open_lth/slurm-setup.sh cifar10

OUTPUT="./outputs/test/online-pointwise.npz"
CKPT_ROOT=$HOME/scratch/open_lth_data/

# VGG-16 lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d
CKPT=(  \
    lottery_b49ffe5e5a5c5bc82fd39df5f148ee0d
)
# CKPT=($(ls $CKPT_ROOT))

REPLICATE=($(seq 1 1 1))

parallel --delay=15 --jobs=1  \
    python -m scripts.gen_metrics  \
        --ckpt_dir=$CKPT_ROOT/{1}"/replicate_"{2}"/level_0/main"  \
        --n_examples=100  \
        --save_file=$OUTPUT  \
    ::: ${CKPT[@]}  \
    ::: ${REPLICATE[@]}  \
