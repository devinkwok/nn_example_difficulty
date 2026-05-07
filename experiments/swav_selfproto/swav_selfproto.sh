#!/bin/bash

DATASET=$1
OUTDIR=$2
SEED=$3
KMEANS=$4

# download swav repo code
if ! [ -d "swav" ]; then
    git clone https://github.com/facebookresearch/swav.git
fi

# put the python script in the swav directory to simplify imports
cp swav_selfproto.py swav/
# do the same with requirements file
cp requirements.txt swav/

cd swav

# download pretrained checkpoint
if [ ! -f "swav_800ep_pretrain.pth.tar" ]; then
    wget "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
fi

# download difficulty code as a submodule
if ! [ -d "nn_example_difficulty" ]; then
    git submodule add https://github.com/devinkwok/nn_example_difficulty.git
fi

# install requirements and copy data
source $HOME/ssetup.sh $DATASET

# set environment variable to data location
export OPEN_LTH_DATASETS="$SLURM_TMPDIR/data/"

# generate intermediates and self supervised prototype scores
python -m swav_selfproto  \
    --dataset=$DATASET  \
    --save_file=$OUTDIR/selfproto-$DATASET-swav.npz  \
    --seed=$SEED  \
    --k=$KMEANS  \
