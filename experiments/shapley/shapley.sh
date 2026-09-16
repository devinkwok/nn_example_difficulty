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
REPLICATE=($(seq $3 1 $4))
EPOCH=(10 20 150 160)

echo $DATASET $DATA $EXP $3 $4

source $HOME/ssetup.sh $DATASET

parallel --delay=2 --jobs=1  \
    python -m experiments.shapley.shapley  \
        --ckpt="$EXP/replicate_{1}/level_0/main/model_ep{2}_it0.pth"  \
        --train  \
        --verbose  \
        --batch_size=64  \
    ::: ${REPLICATE[@]}  \
    ::: ${EPOCH[@]}  \


parallel --delay=2 --jobs=1  \
    python -m experiments.shapley.shapley  \
        --ckpt="$EXP/replicate_{1}/level_0/main/model_ep{2}_it0.pth"  \
        --train  \
        --verbose  \
        --batch_size=64  \
        --include="fc"  \
    ::: ${REPLICATE[@]}  \
    ::: ${EPOCH[@]}  \


# sbatch experiments/shapley/shapley.sh cifar10 lottery_8ae667adac9b690a88b522e85f1388c8 1 100
# sbatch experiments/shapley/shapley.sh cifar100 lottery_9dc2b8dfff54c4da57570d7b391dcccd 1 100
# sbatch experiments/shapley/shapley.sh cifar10 lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5 1 100
# sbatch experiments/shapley/shapley.sh cifar100 lottery_06562e05cda1bf840ee146490815ed82 1 100

# sbatch experiments/shapley/shapley.sh cifar10 lottery_018683f0568061c3a8b71ab3591569b8 20 100
# sbatch experiments/shapley/shapley.sh cifar10 lottery_323890dc4552d843f9a1fcfe3da775bf 50 100
# sbatch experiments/shapley/shapley.sh cifar10 lottery_775f113b9c14d804ba7b473e4ebc835f 70 100
# sbatch experiments/shapley/shapley.sh cifar10 lottery_efe7d47fb664dce78349cfa2ff31da4c 20 100
# sbatch experiments/shapley/shapley.sh cifar100 lottery_a8299123c240af67e983ad684c69ecb9 50 100
