import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from difficulty.utils import stack_metrics, average_metrics, variance_metrics
from difficulty.metrics import *

import sys
sys.path.append("../open_lth")
from open_lth.api import get_device, get_hparams_dict, get_ckpt, \
    get_dataset_hparams, get_dataloader, list_checkpoints


# args
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True, type=Path)
parser.add_argument("--save_dir", required=True, type=Path)
parser.add_argument("--n_examples", default=50000, type=int)
parser.add_argument("--batch_size", default=40, type=int)
parser.add_argument("--train", default=False, action="store_true")
args = parser.parse_args()
print("Generating example difficulty metrics with args:", args)

# get data, model
print(get_hparams_dict(args.ckpt_dir))
if args.batch_size is None:
    batch_size = args.n_examples
dataloader = get_dataloader(get_dataset_hparams(args.ckpt_dir),
                            args.n_examples, args.train, args.batch_size)
loss_fn = nn.CrossEntropyLoss(reduction="none")

def checkpoints():
    for step, ckpt in tqdm(zip(*list_checkpoints(args.ckpt_dir))):
        print("\tloading ckpt", step)
        _, model, _ = get_ckpt(ckpt)
        yield model

def save(subdir, key, step, *save_args, **kwds):
    save_dir = args.save_dir / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / f"{key}_{step.to_str()}.npz", *save_args, **kwds)

steps, _ = list_checkpoints(args.ckpt_dir)

# generate training and test metrics
grands = []
metrics = defaultdict(list)
for model in checkpoints():
    grand, logits = grand_score(model, dataloader, device=get_device(), return_output=True)
    grands.append(grand.detach().cpu())
    labels = torch.cat([y for _, y in dataloader]).to(device=logits.device)
    with torch.no_grad():
        metrics["avgloss"].append(loss_fn(logits.detach(), labels).detach().cpu())
        for k, v in pointwise_metrics(logits.detach(), labels).items():
            metrics[k].append(v)

with torch.no_grad():
    # save losses and grad metrics throughout training
    for step, grand, el2n, loss in zip(steps, grands, metrics["el2n"], metrics["avgloss"]):
        save("gradmetrics", "grand", step, grand.detach().cpu().numpy())
        save("gradmetrics", "el2n", step, el2n.detach().cpu().numpy())
        save("pwmetrics", "loss", step, loss.detach().cpu().numpy())

    # save means and variances at end of training
    metrics = stack_metrics(metrics)
    means = average_metrics(metrics, prefix=None)
    vars = variance_metrics(metrics, prefix=None)
    for k, v in means.items():
        save("pwmetrics", k, steps[-1], mean=v.numpy(), variance=vars[k].numpy())

    # save forget at end of training
    forget = forget_metrics(metrics["acc"])
    for k, v in forget.items():
        save("pwmetrics", k, steps[-1], v.numpy())

# save vog at end of training
classvog = variance_of_gradients(checkpoints(), dataloader, device=get_device())
save("pwmetrics", "classvog", steps[-1], mean="none", variance=classvog.detach().cpu().numpy())
lossvog = variance_of_gradients(checkpoints(), dataloader, device=get_device(), loss_fn=loss_fn)
save("pwmetrics", "lossvog", steps[-1], mean="none", variance=lossvog.detach().cpu().numpy())
