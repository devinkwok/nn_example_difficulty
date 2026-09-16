#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=difficulty-%j.out
#SBATCH --error=difficulty-%j.err


MODEL_CSV=$1
DATASET=$2

DATA_ROOT="../../scratch/data/"
CKPT_ROOT="../../scratch/2023-difficulty/vit_ckpts/"
OUTPUT_ROOT="../../scratch/2023-difficulty/vit_outputs/"
# TMP_DATA_ROOT="/tmp/data/"

# count number of rows in csv, subtracting 1 for header row and 1 for zero-indexing 
N_EVAL=$(cat $MODEL_CSV | wc -l)
N_EVAL="$((N_EVAL - 2))"

echo $MODEL_CSV $DATASET $DATA_ROOT $TMP_DATA_ROOT $CKPT_ROOT $OUTPUT_ROOT $N_EVAL

module load python/3.7
module load pytorch/1.4

# HF token
source ~/skeys.sh

# requirements
source .venv/bin/activate

# # copy imagenet to tmp
# mkdir /tmp/data
# rsync -a $DATA_ROOT/$DATASET $TMP_DATA_ROOT

# evaluate
for i in $(seq 0 $N_EVAL); do
    python -m experiments.vit.vit_difficulty_scores  \
    --ckpt_root=$CKPT_ROOT  \
    --output_root=$OUTPUT_ROOT  \
    --data_root=$DATA_ROOT  \
    --models_csv=$MODEL_CSV  \
    --i=$i  \
    --dataset=$DATASET  \
    --verbose  \

    # --debug  \

done


# run this: for f in selected_models_*.csv; do sbatch difficulty_eval.sh $f imagenet; done
