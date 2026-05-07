#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=retrain_data_pruning-%j.out
#SBATCH --error=retrain_data_pruning-%j.err


REPLICATE=(1 2 3 4 5)
SEED=(16693 6744 26346 28713 10325)

parallel --delay=3  \
    sbatch ./experiments/data_pruning/data_pruning_retrain.sh {1} {2}  \
    ::: ${REPLICATE[@]}  \
    :::+ ${SEED[@]}  \
