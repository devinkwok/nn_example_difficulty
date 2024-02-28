#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

# to run on Mila cluster
source ./open_lth/slurm-setup.sh cifar10 cifar100

REPLICATE=($HOME/scratch/2023-difficulty/metrics/*/)

parallel --delay=15 --jobs=1  \
    python -m scripts.gen_ensemble_metrics  \
        --exp_dir={1}  \
        --ckpt_step=160ep0it  \
        --n_examples=50000  \
        --train  \
        --batch_size=10000  \
    ::: ${REPLICATE[@]}  \
