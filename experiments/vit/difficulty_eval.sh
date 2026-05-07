#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=difficulty-%j.out
#SBATCH --error=difficulty-%j.err

# HF token
source ~/skeys.sh

# requirements
source .venv/bin/activate

# copy imagenet to tmp
mkdir /tmp/data
rsync -a ~/scratch/data/$2 /tmp/data/

# evaluate
for i in $(seq 0 8); do
    python -m difficulty_eval --models_csv $1 --dataset $2 --i $i
done


# run this: for f in selected_models_*.csv; do sbatch difficulty_eval.sh $f imagenet; done