#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=shapley-%j.out
#SBATCH --error=shapley-%j.err

DATASET=$1
EXP=$HOME/scratch/2023-difficulty/ckptmetrics/$2
CKPT=$3
REPLICATE=($(seq 1 1 100))

echo $DATASET $DATA $EXP $CKPT

source $HOME/ssetup.sh $DATASET

parallel --delay=2 --jobs=1  \
    python -m experiments.shapley.shapley  \
        --ckpt="$EXP/replicate_{1}/level_0/main/model_$CKPT.pth"  \
        --train  \
        --verbose  \
        --batch_size=64  \
    ::: ${REPLICATE[@]}  \


# sbatch experiments/shapley/shapley.sh cifar10 lottery_8ae667adac9b690a88b522e85f1388c8 ep10_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_8ae667adac9b690a88b522e85f1388c8 ep20_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_8ae667adac9b690a88b522e85f1388c8 ep150_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_8ae667adac9b690a88b522e85f1388c8 ep160_it0

# sbatch experiments/shapley/shapley.sh cifar100 lottery_9dc2b8dfff54c4da57570d7b391dcccd ep10_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_9dc2b8dfff54c4da57570d7b391dcccd ep20_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_9dc2b8dfff54c4da57570d7b391dcccd ep150_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_9dc2b8dfff54c4da57570d7b391dcccd ep160_it0

# sbatch experiments/shapley/shapley.sh cifar10 lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5 ep10_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5 ep20_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5 ep150_it0
# sbatch experiments/shapley/shapley.sh cifar10 lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5 ep160_it0

# sbatch experiments/shapley/shapley.sh cifar100 lottery_06562e05cda1bf840ee146490815ed82 ep10_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_06562e05cda1bf840ee146490815ed82 ep20_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_06562e05cda1bf840ee146490815ed82 ep150_it0
# sbatch experiments/shapley/shapley.sh cifar100 lottery_06562e05cda1bf840ee146490815ed82 ep160_it0