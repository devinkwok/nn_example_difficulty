#!/bin/bash

CONFIG_FILE=$1
DATASET=$2
NTASKS=${3:-1}  # default to one task


chmod +x ./scripts/difficulty/train_adam.sh
source $HOME/ssetup-uv.sh $DATASET

# export OPEN_LTH_ROOT="$SLURM_TMPDIR/open_lth_data/"
export OPEN_LTH_ROOT="$SSETUP_OUTPUT_DIR/open_lth_data/"
export OPEN_LTH_DATASETS="$SLURM_TMPDIR/data/"
mkdir -p $OPEN_LTH_ROOT


srun -l --ntasks $NTASKS --output=../slurm-%j-%t.out --multi-prog ./scripts/difficulty/$CONFIG_FILE.conf
