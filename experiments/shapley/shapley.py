import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.environ['HOME'], "lib", "open_lth"))
sys.path.append("src")
import torch.nn as nn

from difficulty.metrics.gradient import gradient_product_scores
from difficulty.utils import detach_tensors, Stopwatch, match_key
from open_lth import api


def get_by_model_name(model_hparams, model_to_value_dict, failure_msg):
    model_name = model_hparams.model_name
    for k, v in model_to_value_dict.items():
        if k in model_name:
            return v
    warnings.warn(f"Unknown model {model_name}: {failure_msg}")
    return None

def score_file(directory, score_name, ep_it):
    return directory / f"{score_name}_{ep_it}.npz"

def load_scores(directory, score_names, ep_it, device):
    if not directory.exists():
        return None
    scores = {}
    print(f"\nLoading from {directory}...")
    for score_name in score_names:
        file = score_file(directory, score_name, ep_it)
        if not file.exists():
            return None
        score = np.load(file)["arr_0"]
        try:
            score = torch.tensor(score, device=device)
        except TypeError:
            pass  # if type is string don't convert
        scores[score_name] = score
    return scores

def save_scores(directory, scores, ep_it):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {directory}...")
    for score_name, score in scores.items():
        print(f"... saving {score_name}")
        np.savez(score_file(directory, score_name, ep_it), score)


def generate_scores(args):
    # lottery_*/replicate_*/level_0/main/model_ep_it.pth
    ep_it = args.ckpt.stem.split("model_")[1]
    save_dir = args.ckpt.parent / "ckptmetrics"
    val_samples = [int(i) for i in args.val_samples.split(",")]
    include = args.include.split(",") if len(args.include) > 0 else None
    exclude = args.exclude.split(",") if len(args.exclude) > 0 else None

    score_names = [f"tracin{n}" for n in val_samples] + [f"tracin{n}partial" for n in val_samples] + ["grandpartial"]
    scores = load_scores(save_dir, score_names, ep_it, args.device)

    if scores is None:
        stopwatch = Stopwatch(verbose=args.verbose)

        (model_hparams, _), model, _ = api.get_ckpt(args.ckpt)
        dataset_hparams = api.get_dataset_hparams(args.ckpt)
        dataloader = api.get_dataloader(dataset_hparams, train=args.train, batch_size=args.batch_size)
        val_dataloader = api.get_dataloader(dataset_hparams, train=(not args.train), batch_size=args.batch_size)
        # representation_layer = get_by_model_name(
        #     model_hparams, PROTO_REPRESENTATION_LAYER, "using last layer for prototypes")

        warnings.warn(f"Generating scores...")
        with stopwatch:
            scores = gradient_product_scores(
                model,
                dataloader,
                val_dataloader,
                val_samples,
                include=include,
                exclude=exclude,
                device=args.device,
                to_cpu=True,
                to_numpy=True,
            )
        scores["NEWgrand"] = scores["grand"]
        del scores["grand"]
        
        # save scores per replicate
        save_scores(save_dir, scores, ep_it)
    print(f"Finished {args.ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--val_samples', default="1,100,10000", type=str)
    parser.add_argument('--include', default="", type=str)
    parser.add_argument('--exclude', default="", type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--verbose', default=False, action="store_true")

    args = parser.parse_args()
    generate_scores(args)
