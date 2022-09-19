import argparse
from pathlib import Path
import numpy as np
import torch

from difficulty.model.eval import compute_logits_for_checkpoints
from difficulty.model.open_lth_utils import get_iterative_magnitude_pruning_checkpoints, get_training_checkpoints, get_hparams, get_dataset, get_model
from difficulty.utils import pointwise_metrics, add_prefix, avg_metrics, last_metrics, save_metrics_dict, train_forget_metrics, perturb_forget_metrics


# args
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_root", required=True, type=Path)
parser.add_argument("--ckpt", required=True, type=Path)
parser.add_argument("--replicate", required=True, type=int)
parser.add_argument("--n_examples", default=10000, type=int)
parser.add_argument("--train", default=False, type=Path)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--batch_size", default=None, type=int)
parser.add_argument("--data_root", default=(Path.home() / "open_lth_datasets"), type=Path)
parser.add_argument("--out_dir", default=Path("./outputs"), type=Path)
args = parser.parse_args()
print("Generating example difficulty metrics with args:", args)

# get data, model
ckpt_dir = args.ckpt_root / args.ckpt
hparams = get_hparams(ckpt_dir)
print(hparams)
model = get_model(hparams, args.device)
if args.batch_size is None:
    batch_size = args.n_examples
dataset = get_dataset(hparams, args.train, args.data_root)
dataset = torch.utils.data.Subset(dataset, np.arange(args.n_examples))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
labels = np.array([y for _, y in dataset])

# generate training and test metrics
train_iter, train_ckpts = get_training_checkpoints(hparams, ckpt_dir, args.replicate)
train_logits = compute_logits_for_checkpoints(train_ckpts, model, dataloader, device=args.device)
train_pointwise = pointwise_metrics(train_logits, labels)
train_forget = train_forget_metrics(train_logits, labels)

# generate pruning logits
prune_fraction, prune_ckpts = get_iterative_magnitude_pruning_checkpoints(
                hparams, ckpt_dir, args.replicate)
prune_logits = compute_logits_for_checkpoints(prune_ckpts, model, dataloader, device=args.device)
prune_pointwise = pointwise_metrics(prune_logits, labels)
prune_forget = perturb_forget_metrics(prune_logits, labels)

# add prefix, average over training/pruning where applicable
metrics = {
    **add_prefix("train", avg_metrics(train_pointwise)),
    **add_prefix("train", train_forget),
    **add_prefix("test", last_metrics(train_pointwise)),
    **add_prefix("prune", avg_metrics(prune_pointwise)),
    **add_prefix("prune", prune_forget),
}
#TODO prediction_depth filter for the following layers:
# MLP: dense, softmax
# VGG: conv, softmax
# ResNet: initial bn, block sum, softmax

# save metrics
save_file = save_metrics_dict(metrics, args.out_dir, args.ckpt, args.replicate)
print(f"Metrics saved to {save_file}")
