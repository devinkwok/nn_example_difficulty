import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from difficulty.model.eval import evaluate_model, get_labels
from difficulty.utils import pointwise_metrics, add_prefix, avg_metrics, last_metrics, train_forget_metrics, perturb_forget_metrics
from difficulty.metrics import forgetting_events, zero_one_accuracy

import sys
sys.path.append("open_lth")
from open_lth.api import get_device, get_hparams_dict, get_ckpt, \
    get_dataset_hparams, get_dataloader, list_checkpoints


# args
parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", required=True, type=Path)
parser.add_argument("--save_file", required=True, type=Path)
parser.add_argument("--train", default="train", type=str)
parser.add_argument("--n_examples", default=10000, type=int)
parser.add_argument("--batch_size", default=None, type=int)
args = parser.parse_args()
print("Generating example difficulty metrics with args:", args)

# get data, model
train = (args.train == "train")
print(get_hparams_dict(args.ckpts))
if args.batch_size is None:
    batch_size = args.n_examples
dataloader = get_dataloader(get_dataset_hparams(args.ckpts),
                            args.n_examples, train, args.batch_size)

# generate training and test metrics
train_iter, train_ckpts = list_checkpoints(args.ckpts)
logits = []
for ckpt in tqdm(train_ckpts):
    _, model, _ = get_ckpt(ckpt)
    logits.append(evaluate_model(model, dataloader, device=get_device()))
logits = np.stack(logits, axis=0)

labels = get_labels(dataloader)
train_pointwise = pointwise_metrics(logits, labels)

metrics = {
    "ep_it": [str(x) for x in train_iter],
    **train_pointwise,
}

# train_forget = train_forget_metrics(logits, labels)

# # generate pruning logits
# prune_fraction, prune_ckpts = get_iterative_magnitude_pruning_checkpoints(
#                 hparams, args.ckpts)
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

# save metrics
args.save_file.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.save_file, **metrics)
print(f"Metrics saved to {args.save_file}")
