import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from difficulty.model.eval import evaluate_model
from difficulty.utils import pointwise_metrics, add_prefix, avg_metrics, last_metrics, train_forget_metrics, perturb_forget_metrics
from difficulty.metrics import OnlineVariance

import sys
sys.path.append("open_lth")
from open_lth.api import get_device, get_hparams_dict, get_ckpt, \
    get_dataset_hparams, get_dataloader, list_checkpoints


# args
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True, type=Path)
parser.add_argument("--save_file", required=True, type=Path)
parser.add_argument("--n_examples", default=10000, type=int)
parser.add_argument("--batch_size", default=None, type=int)
parser.add_argument("--train", default=False, action="store_true")
args = parser.parse_args()
print("Generating example difficulty metrics with args:", args)

# get data, model
print(get_hparams_dict(args.ckpt_dir))
if args.batch_size is None:
    batch_size = args.n_examples
dataloader = get_dataloader(get_dataset_hparams(args.ckpt_dir),
                            args.n_examples, args.train, args.batch_size)

# generate training and test metrics
online_metrics = defaultdict(OnlineVariance)
for step, ckpt in tqdm(zip(*list_checkpoints(args.ckpt_dir))):
    print("\tloading ckpt", step)
    _, model, _ = get_ckpt(ckpt)
    logits, labels, acc, loss = evaluate_model(
        model, dataloader, device=get_device(), loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    for k, v in pointwise_metrics(logits, labels).items():
        online_metrics[k] = online_metrics[k].add(v, ep=step.ep, it=step.it, total_it=step.iteration)

metrics = {}
for k, v in online_metrics.items():
    print("\tsummarizing metric", k)
    metrics["epoch"] = v.metadata["ep"]
    metrics[f"{k}_mean"] = v.mean.get().detach().cpu().numpy()
    metrics[f"{k}_std"] = torch.sqrt(v.get()).detach().cpu().numpy()

# save metrics
args.save_file.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.save_file, **metrics)
print(f"Metrics saved to {args.save_file}")



# train_forget = train_forget_metrics(logits, labels)

# # generate pruning logits
# prune_fraction, prune_ckpts = get_iterative_magnitude_pruning_checkpoints(
#                 hparams, args.ckpt_dir)
# prune_logits = compute_logits_for_checkpoints(prune_ckpts, model, dataloader, device=device)
# prune_pointwise = pointwise_metrics(prune_logits, labels)
# prune_forget = perturb_forget_metrics(prune_logits, labels)

#TODO prediction_depth filter for the following layers:
# MLP: dense, softmax
# VGG: conv, softmax
# ResNet: initial bn, block sum, softmax
# intermediate_activations = 
# prediction_depth(intermediate_activations, consensus_labels, include=["conv", "softmax"])

# add prefix, average over training/pruning where applicable
# metrics = {
#     **add_prefix("train", avg_metrics(train_pointwise)),
#     **add_prefix("train", train_forget),
#     # **add_prefix("test", last_metrics(train_pointwise)),
#     # **add_prefix("prune", avg_metrics(prune_pointwise)),
#     # **add_prefix("prune", prune_forget),
#     # "is_train": get_train_test_split(hparams, train, args.n_examples),
# }
