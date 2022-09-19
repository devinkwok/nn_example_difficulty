#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err

module load python/3.7
module load pytorch/1.4

if ! [ -d "$SLURM_TMPDIR/env/" ]; then
    virtualenv $SLURM_TMPDIR/env/
    source $SLURM_TMPDIR/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $SLURM_TMPDIR/env/bin/activate
fi

MODEL=(cifar_resnet_20 cifar_vgg_16 cifar_lenet_1024_512_128)
REPLICATE=($(seq 1 1 5))
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
    python open_lth.py lottery  \
        --default_hparams={1}  \
        --replicate={2}  \
        --levels=20  \
        --rewinding_steps=160ep  \
        --save_every_n_epochs=1  \
  ::: ${MODEL[@]}  \
  ::: ${REPLICATE[@]}  \
