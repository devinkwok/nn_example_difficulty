import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.environ['HOME'], "lib", "open_lth"))
sys.path.append(os.path.join(os.environ['HOME'], "lib", "nn_example_difficulty", "src"))
import torch.nn as nn

from difficulty.utils import detach_tensors, Stopwatch
from open_lth import api

# PROTO_REPRESENTATION_LAYER = {
#     "cifar_resnet": "fc.in",
#     "cifar_vgg": "fc.in",
# }


def gradient_vector(model):
    grad = torch.cat([x.grad.detach().flatten() for x in model.parameters() if hasattr(x, "grad")])
    return grad


def get_random_samples(dataloader, n_samples):
    val_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=n_samples, shuffle=True)
    batch = next(iter(val_dataloader))
    return [batch]


def _get_val_gradient(model, dataloader, device):
    model = model.to(device=device)
    model.eval()  # freeze batchnorm
    criterion = nn.CrossEntropyLoss(reduction="mean")

    model.zero_grad()
    for x, y in tqdm(dataloader):
        y_pred = model.forward(x.to(device=device))
        loss = criterion(y_pred, y.to(device=device))
        loss.backward()

    # divide by number of batches, as reduction="mean" already divides by samples per batch
    val_grad = gradient_vector(model) / len(dataloader)
    return val_grad


def get_val_gradient(model, dataloader, device, n_samples: Optional[int]=None):
    if n_samples is not None and n_samples < len(dataloader.dataset):
        dataloader = get_random_samples(dataloader, n_samples)
    val_gradient = _get_val_gradient(model, dataloader, device)
    return val_gradient


def brute_force_grad_dotprod(model, dataloader, product_vectors, loss_fn, device):
    model = model.to(device=device)
    model.eval()  # freeze batchnorm

    products = []
    for x, y in tqdm(dataloader):
        for i in range(len(x)):  # backprop one example at a time
            model.zero_grad()
            y_pred = model.forward(x[i:i+1].to(device=device))
            loss = loss_fn(y_pred, y[i:i+1].to(device=device))
            loss.backward()
            grad = gradient_vector(model)  # (W,)
            products.append(grad @ product_vectors.T)  # (1, W) @ (P, W)^T = (P,)
    return torch.stack(products, dim=0)  # (N, P)


def tracin(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    val_samples: list[int],
    loss_fn=None,
    device: str="cpu",
    to_cpu=True,
    to_numpy=False,
    dtype: Union[str, torch.dtype]=torch.float64,
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    val_grads = [get_val_gradient(model, val_dataloader, device, n_samples=n) for n in val_samples]
    dot_prod = brute_force_grad_dotprod(model, dataloader, torch.stack(val_grads, dim=0), loss_fn, device)
    dot_prod = dot_prod.to(dtype=dtype)
    scores = {f"tracin{n}": dot_prod[:, i] for i, n in enumerate(val_samples)}

    scores = detach_tensors(scores, to_cpu=to_cpu, to_numpy=to_numpy)
    return scores


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

    scores = load_scores(save_dir, [f"tracin{n}" for n in val_samples], ep_it, args.device)
    if scores is None:
        stopwatch = Stopwatch(verbose=args.verbose)

        (model_hparams, _), model, _ = api.get_ckpt(args.ckpt)
        dataset_hparams = api.get_dataset_hparams(args.path_to_data_hparam)
        dataloader = api.get_dataloader(dataset_hparams, train=args.train, batch_size=args.batch_size)
        val_dataloader = api.get_dataloader(dataset_hparams, train=(not args.train), batch_size=args.batch_size)
        # representation_layer = get_by_model_name(
        #     model_hparams, PROTO_REPRESENTATION_LAYER, "using last layer for prototypes")

        warnings.warn(f"Generating scores...")
        with stopwatch:
            scores = tracin(model, dataloader, val_dataloader, val_samples)
        # save scores per replicate
        save_scores(save_dir, scores, ep_it)
    print(f"Finished {args.ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--path_to_data_hparam', required=True, type=Path)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--val_samples', default="1,100,10000", type=str)
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()
    generate_scores(args)
