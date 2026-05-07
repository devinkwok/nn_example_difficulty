import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.environ['HOME'], "lib", "open_lth"))
sys.path.append(os.path.join(os.environ['HOME'], "lib", "nn_example_difficulty", "src"))
from open_lth import api
from difficulty.metrics import ensemble_metrics, representation_metrics, grand_score
from difficulty.model.eval import find_intermediate_layers
from difficulty.utils import Stopwatch


# see Appendix A.5 in Baldock et al. "Deep Learning Through the Lens of Example Difficulty"
PD_LAYERS_TO_INCLUDE = {
    "cifar_resnet": ["bn.out", ".relu2.in"],  # after initial norm, each sum operation (i.e. shortcut), softmax
    "cifar_vgg": [".conv.out"],  # after convolutions, softmax
}  # note: for softmax, set pd_append_softmax=True in representation_metrics

def get_by_model_name(hparams_dict, model_to_value_dict, failure_msg):
    model_name = api.get_model_hparams(hparams_dict).model_name
    for k, v in model_to_value_dict.items():
        if k in model_name:
            return v
    warnings.warn(f"Unknown model {model_name}: {failure_msg}")
    return None

def find_replicates(exp_path, ep_it):
    # go through replicates
    for replicate in exp_path.glob("replicate_*"):
        # if ep_it isn't available in replicate, this will raise exception
        ckpt_path = api.find_ckpt_by_it(replicate, ep_it)
        # load specific checkpoint
        (_, _), model, _ = api.get_ckpt(ckpt_path)
        yield replicate, model

def replicates_model_only(exp_path, ep_it):
    for _, model in tqdm(find_replicates(exp_path, ep_it)):
        yield model

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
        print(f"... loading {file.name}")
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

def generate_scores(exp_path, ep_it, dataloader, device, include_layers, verbose):
    stopwatch = Stopwatch("ensemble_metrics", verbose=args.verbose)

    # eval replicates once to get consensus label
    # save to exp root
    warnings.warn(f"Loading ensemble scores for {exp_path}")
    ensemble_save_dir = exp_path / "ensemble_metrics"
    ensemble_scores = load_scores(ensemble_save_dir, ["ddd", "consensuslabel", "allacc"], ep_it, device)
    if ensemble_scores is None:
        warnings.warn(f"Generating ensemble scores...")
        with stopwatch:
            ensemble_scores = ensemble_metrics(replicates_model_only(exp_path, ep_it), dataloader, device=device)
        save_scores(ensemble_save_dir, ensemble_scores, ep_it)

    pdlayers = load_scores(ensemble_save_dir, ["pdlayers"], ep_it, device=device)
    if pdlayers is None:
        model = next(iter(replicates_model_only(exp_path, ep_it)))
        layers = find_intermediate_layers(model, next(iter(dataloader))[0].shape[1:], device=args.device, include=include_layers)
        pdlayers = {"pdlayers": layers}
        save_scores(ensemble_save_dir, pdlayers, ep_it)
    print(f"Prediction depth will use layers: {pdlayers['pdlayers']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, type=Path)
    parser.add_argument('--ep_it', required=True, type=str)
    parser.add_argument('--path_to_data_hparam', required=True, type=Path)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--batch_size', default=5000, type=int)
    parser.add_argument('--device', default="cuda", type=str)  # only affects KNN of self-supervised prototypes
    parser.add_argument('--seed', default=42, type=int)  # only affects KNN of self-supervised prototypes
    args = parser.parse_args()

    dataset_hparams = api.get_dataset_hparams(args.path_to_data_hparam)
    dataloader = api.get_dataloader(dataset_hparams, train=args.train, batch_size=args.batch_size)
    n_classes = api.num_classes(dataset_hparams)

    if args.debug:  # only use first 2 batches in debug mode
        warnings.warn("DEBUG mode on!!!")
        xs, ys = [], []
        for _, (x, y) in zip(range(2), dataloader):
            xs.append(x)
            ys.append(y)
        debug_dataset = torch.utils.data.TensorDataset(torch.cat(xs), torch.cat(ys))
        dataloader = torch.utils.data.DataLoader(debug_dataset)

    hparams = api.get_hparams_dict(args.exp)
    # skip if experiment has no metrics
    if hparams["training_hparams"]["metrics_n_train"] == 0 and hparams["training_hparams"]["metrics_n_test"] == 0:
        raise RuntimeError(f"No metrics found in {args.exp}")

    # look at type of network and get the correct subset of layers for prediction depth
    include_layers = get_by_model_name(
        hparams, PD_LAYERS_TO_INCLUDE, "using all layers for prediction depth")

    generate_scores(args.exp, args.ep_it, dataloader, args.device, include_layers, args.verbose)
