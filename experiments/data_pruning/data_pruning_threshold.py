from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
import torch

from difficulty.metrics import rank


# args
parser = argparse.ArgumentParser()
parser.add_argument("--score_file", required=True, type=Path)
parser.add_argument("--score_key", default=None, type=str)
parser.add_argument("--subset_fraction", default=0.5, type=float)
parser.add_argument("--subset_offset", default=None, type=float)
parser.add_argument("--subset_smallest", default=False, action="store_true")
parser.add_argument("--subset_invert", default=False, action="store_true")
parser.add_argument("--save_file", default=None, type=Path)
args = parser.parse_args()
print("Creating data subset:", args)

# load .npz score file
scores = np.load(args.score_file)
# if key not specified, take first key in score file
key = next(iter(scores.keys())) if args.score_key is None else args.score_key
print("Array key:", key)
ranks = rank(torch.tensor(scores[key])).numpy()
n_examples = len(ranks)

# data subset is between subset_offset and subset_offset + fraction_keep
if args.subset_offset is None:
    subset_offset = 0 if args.subset_smallest else 1 - args.subset_fraction
else:
    subset_offset = args.subset_offset
subset_end = subset_offset + args.subset_fraction
assert subset_offset >= 0 and args.subset_fraction > 0 and subset_end <= 1
start = int(subset_offset * n_examples)
end = int(subset_end * n_examples)
# subset never exceeds subset_fraction
if (end - start) / n_examples > args.subset_fraction:
    start += 1

# save array of example indices as subset
subset_idx = np.logical_and(ranks >= start, ranks < end)
if args.subset_invert:
    subset_idx = np.logical_not(subset_idx)
subset = np.arange(n_examples)[subset_idx]
args.save_file.parent.mkdir(parents=True, exist_ok=True)
np.save(args.save_file, subset)
print("Saved to", args.save_file)
