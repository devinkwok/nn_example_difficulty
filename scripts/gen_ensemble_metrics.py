import argparse
from pathlib import Path
import numpy as np
import torch

from difficulty.metrics import ensemble_metrics

import sys
sys.path.append("../open_lth")
from open_lth.api import get_device, get_ckpt, \
    get_dataset_hparams, get_dataset, get_dataloader, list_checkpoints


# args
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", required=True, type=Path)
parser.add_argument("--ckpt_step", required=True, type=str)
parser.add_argument("--level", default="0", type=str)
parser.add_argument("--branch", default="main", type=str)
parser.add_argument("--save_dir", default=None, type=Path)
parser.add_argument("--n_examples", default=50000, type=int)
parser.add_argument("--batch_size", default=None, type=int)
parser.add_argument("--train", default=False, action="store_true")
args = parser.parse_args()
print("Generating example difficulty metrics with args:", args)

# list model checkpoints
def checkpoints():
    for replicate in args.exp_dir.glob("replicate_*"):
        for step, file in zip(*list_checkpoints(replicate / f"level_{args.level}" / args.branch)):
            if step.to_str() == args.ckpt_step:
                yield file

# get dataset
dataset_hparams = get_dataset_hparams(next(checkpoints()))
print(dataset_hparams)
if args.batch_size is None:
    batch_size = args.n_examples
dataset_obj = get_dataset(dataset_hparams)
dataloader = get_dataloader(dataset_hparams, args.n_examples, args.train, args.batch_size)

# generate metrics
def models():
    for ckpt in checkpoints():
        _, model, _ = get_ckpt(ckpt)
        yield model

with torch.no_grad():
    metrics = ensemble_metrics(models(), dataloader, dataset_obj.num_classes(), device=get_device())

# save metrics
save_dir = args.exp_dir if args.save_dir is None else args.save_dir
save_dir.mkdir(parents=True, exist_ok=True)
for k, v in metrics.items():
    np.savez(save_dir / f"{k}_{args.ckpt_step}.npz", v.detach().cpu().numpy())
