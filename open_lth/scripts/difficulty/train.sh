#!/bin/bash

SCRIPT_NAME=$1
MODEL=$2
DATASET=$3
START=$4
END=$5
NTASKS=${6:-1}  # default to one task


chmod +x ./scripts/difficulty/$SCRIPT_NAME.sh
source $HOME/ssetup-uv.sh $DATASET
# source $HOME/ssetup-uv.sh cifar10 cinic10

# export OPEN_LTH_ROOT="$SLURM_TMPDIR/open_lth_data/"
export OPEN_LTH_ROOT="$SSETUP_OUTPUT_DIR/open_lth_data/"
export OPEN_LTH_DATASETS="$SLURM_TMPDIR/data/"
mkdir -p $OPEN_LTH_ROOT


for REPLICATE in $(seq $START $NTASKS $END); do
    srun -l --ntasks $NTASKS --output=../slurm-%j-%t.out ./scripts/difficulty/$SCRIPT_NAME.sh $MODEL $DATASET $REPLICATE
done
