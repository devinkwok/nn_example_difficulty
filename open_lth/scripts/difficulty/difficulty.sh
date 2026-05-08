#!/bin/bash


## standard
sbatch --ntasks=3 --ntasks-per-node=3 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_vgg_16 cifar10 1 100 3
sbatch --mem-per-gpu=48G --ntasks=6 --ntasks-per-node=6 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20 cifar10 1 100 6
sbatch --mem-per-gpu=48G --ntasks=4 --ntasks-per-node=4 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_32 cifar10 1 100 4
sbatch --mem-per-gpu=48G --ntasks=4 --ntasks-per-node=4 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20_64 cifar10 1 100 4

## standard cifar100
sbatch --ntasks=3 --ntasks-per-node=3 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_vgg_16 cifar100 1 100 3
sbatch --mem-per-gpu=48G --ntasks=6 --ntasks-per-node=6 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20 cifar100 1 100 6
sbatch --mem-per-gpu=48G --ntasks=4 --ntasks-per-node=4 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20_64 cifar100 1 100 4

## adam
sbatch --mem-per-gpu=48G --ntasks=4 --ntasks-per-node=4 --cpus-per-task=2 ./scripts/difficulty/train.sh train_adam cifar_resnet_20 cifar10 1 100 4

## cinic10 excluding cifar10
sbatch --mem-per-gpu=48G --ntasks=4 --ntasks-per-node=4 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20 cinic10 1 100 4
sbatch --mem-per-gpu=48G --ntasks=2 --ntasks-per-node=2 --cpus-per-task=2 ./scripts/difficulty/train.sh train_standard cifar_resnet_20 cinic10nocifartrain 9990 9991 2


## low lr
sbatch --mem-per-gpu=48G --ntasks=6 --ntasks-per-node=6 --cpus-per-task=2 ./scripts/difficulty/train.sh train_lowlr cifar_resnet_20 cifar10 1 100 6

## cosine instead of step lr schedule
sbatch --mem-per-gpu=48G --ntasks=6 --ntasks-per-node=6 --cpus-per-task=2 ./scripts/difficulty/train.sh train_onecycle cifar_resnet_20 cifar10 1 6 6
# test cosine lr schedule
sbatch --mem-per-gpu=16G --cpus-per-task=2 ./scripts/difficulty/single_train.sh train_onecycle cifar_resnet_20 cifar10 1 2

# sketch of next steps:
# python collate_metrics.py --save-dir $TARGET
# for SUBDIR in $TARGET; do
#   for CKPT in $CKPTS; do
#     sbatch ckptmetrics.sh $DATAROOT $SUBDIR $CKPT
#   done
# done

