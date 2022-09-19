#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

# to run on Mila cluster
module load python/3.7
module load pytorch/1.4

if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    # set up python environment
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    # install dependencies
    pip install --upgrade pip
    pip install -r "./requirements.txt"
else
    source $SLURM_TMPDIR/env/bin/activate
fi

OUTPUT="./outputs"
EXAMPLES=10000

CKPT_ROOT=$HOME/scratch/open_lth_data/

# VGG-16 lottery_06e3ceea2dae7621529556ef969cf803
# ResNet-20 lottery_938ede76e304643f5466ed419261dc65
# MLP-3 lottery_ab596c041ffd39d837f0a60d39d86c72

CKPT=(  \
    lottery_06e3ceea2dae7621529556ef969cf803  \
    lottery_938ede76e304643f5466ed419261dc65  \
    lottery_ab596c041ffd39d837f0a60d39d86c72  \
)

REPLICATE=($(seq 1 1 5))

parallel --delay=15 --linebuffer --jobs=3  \
    python -m gen_metrics  \
        --ckpt_root=$CKPT_ROOT  \
        --ckpt={1}  \
        --replicate={2}  \
        --n_examples=$EXAMPLES  \
        --out_dir=$OUTPUT  \
    ::: ${CKPT[@]}  \
    ::: ${REPLICATE[@]}  \
