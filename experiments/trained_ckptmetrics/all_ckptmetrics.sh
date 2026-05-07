#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=gen-metrics-%j.out
#SBATCH --error=gen-metrics-%j.err

EXPROOT="$HOME/scratch/2023-difficulty/ckptmetrics/"


DATASET=cifar10
# DIRS=(  \
#     lottery_cf40a5a8bbe4f285b8fe317c1aa95dd5  \
#     lottery_8ae667adac9b690a88b522e85f1388c8  \
#     lottery_323890dc4552d843f9a1fcfe3da775bf  \
#     lottery_018683f0568061c3a8b71ab3591569b8  \
#     lottery_775f113b9c14d804ba7b473e4ebc835f  \
#     lottery_efe7d47fb664dce78349cfa2ff31da4c  \
# )

DATASET=cifar100
DIRS=(  \
    lottery_06562e05cda1bf840ee146490815ed82  \
    lottery_9dc2b8dfff54c4da57570d7b391dcccd  \
    lottery_a8299123c240af67e983ad684c69ecb9  \
)

# CKPTS=(ep0_it0 ep10_it0 ep20_it0 ep150_it0 ep160_it0)
CKPTS=(ep10_it0 ep20_it0 ep150_it0 ep160_it0)
# CKPTS=(ep150_it0 ep160_it0)

parallel --delay=5 --jobs=1  \
    sbatch ckptmetrics.sh $DATASET $EXPROOT/{1}/replicate_1/ $EXPROOT/{1} {2}  \
    ::: ${DIRS[@]}  \
    ::: ${CKPTS[@]}  \
