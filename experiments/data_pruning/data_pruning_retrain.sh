#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=data_pruning_retrain-%j.out
#SBATCH --error=data_pruning_retrain-%j.err


MODEL=cifar_resnet_20
DATASET=cifar10
REPLICATE=$1
SEED=$2
FRACTION="0.1"
OFFSET=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
SUBSET=$HOME/scratch/2023-difficulty/metrics/lottery_8ae667adac9b690a88b522e85f1388c8/data_pruning

source ../2023-difficulty/open_lth/slurm-setup.sh cifar10
cd ../2023-difficulty/open_lth/

# with average score
parallel --delay=15 --jobs=3  \
    python open_lth.py lottery_branch retrain  \
    --default_hparams=$MODEL  \
    --dataset_name=$DATASET  \
    --replicate=$REPLICATE  \
    --data_order_seed=$SEED  \
    --levels=0  \
    --training_steps=160ep  \
    --save_ckpt_steps="0ep,10ep,20ep,150ep,160ep"  \
    --metrics_n_train=50000  \
    --pointwise_metrics_steps="1ep-160ep"  \
    --pointwise_metrics_batch_size=1000  \
    --grad_metrics_steps="10ep,20ep"  \
    --grad_metrics_batch_size=40  \
    --batch_forget_track  \
        --retrain_d_subset_file=$SUBSET/"el2n_20ep0it_$FRACTION-"{1}"_avg.npy"  \
        --retrain_d_dataset_name=$DATASET  \
        --retrain_d_batch_size=128  \
        --retrain_t_optimizer_name='sgd'  \
        --retrain_t_momentum=0.9  \
        --retrain_t_milestone_steps='80ep,120ep'  \
        --retrain_t_lr=0.1  \
        --retrain_t_gamma=0.1  \
        --retrain_t_weight_decay=1e-4  \
        --retrain_t_training_steps='160ep'  \
        --retrain_t_warmup_steps=1ep  \
        --start_at=init  \
    ::: ${OFFSET[@]}  \

# with per-replicate score
parallel --delay=15 --jobs=3  \
    python open_lth.py lottery_branch retrain  \
    --default_hparams=$MODEL  \
    --dataset_name=$DATASET  \
    --replicate=$REPLICATE  \
    --data_order_seed=$SEED  \
    --levels=0  \
    --training_steps=160ep  \
    --save_ckpt_steps="0ep,10ep,20ep,150ep,160ep"  \
    --metrics_n_train=50000  \
    --pointwise_metrics_steps="1ep-160ep"  \
    --pointwise_metrics_batch_size=1000  \
    --grad_metrics_steps="10ep,20ep"  \
    --grad_metrics_batch_size=40  \
    --batch_forget_track  \
        --retrain_d_subset_file=$SUBSET/"el2n_20ep0it_$FRACTION-"{1}"_replicate_$REPLICATE.npy"  \
        --retrain_d_dataset_name=$DATASET  \
        --retrain_d_batch_size=128  \
        --retrain_t_optimizer_name='sgd'  \
        --retrain_t_momentum=0.9  \
        --retrain_t_milestone_steps='80ep,120ep'  \
        --retrain_t_lr=0.1  \
        --retrain_t_gamma=0.1  \
        --retrain_t_weight_decay=1e-4  \
        --retrain_t_training_steps='160ep'  \
        --retrain_t_warmup_steps=1ep  \
        --start_at=init  \
    ::: ${OFFSET[@]}  \
