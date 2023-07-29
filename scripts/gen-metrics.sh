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

python -m scripts.gen_from_ckpts  \
    --ckpt_dir=$HOME/scratch/open_lth_data/lottery_5b558810ec90bac2036122744c22e4fe/replicate_1/level_0/main/  \
    --save_dir=./outputs/test-metrics/  \
    --n_examples=50000  \
    --train  \
    --batch_size=40  \
