#!/bin/bash
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=scores-%j.out
#SBATCH --error=scores-%j.err

# requirements
source .venv/bin/activate

# evaluate
for i in $(seq 0 7); do
    python -m vit.compute_scores_limited --models_csv $1 --dataset $2 --i $i
done


# run this: for f in vit/selected_models_*.csv; do sbatch compute_scores_limited.sh $f imagenet; done
