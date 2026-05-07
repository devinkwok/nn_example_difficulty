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
from difficulty.metrics import representation_metrics, grand_score
from difficulty.utils import Stopwatch


# see Appendix B in Sorscher et al. "Beyond Neural Scaling Laws"
PROTO_REPRESENTATION_LAYER = {
    "cifar_resnet": "fc.in",
    "cifar_vgg": "fc.in",
}  # note: fc.in is output of pooling layer, to match SWaV output which is after avgpool

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
    ensemble_save_dir = args.ckpt.parent.parent.parent.parent / "ensemble_metrics"

    scores = load_scores(save_dir, ["grand", "pd", "selfproto", "proto"], ep_it, args.device)
    if scores is None:
        stopwatch = Stopwatch(verbose=args.verbose)

        (model_hparams, _), model, _ = api.get_ckpt(args.ckpt)
        consensus_labels = load_scores(ensemble_save_dir, ["consensuslabel"], ep_it, args.device)["consensuslabel"]
        layers = load_scores(ensemble_save_dir, ["pdlayers"], ep_it, device=args.device)["pdlayers"]
        dataset_hparams = api.get_dataset_hparams(args.path_to_data_hparam)
        dataloader = api.get_dataloader(dataset_hparams, train=args.train, batch_size=args.batch_size)
        n_classes = api.num_classes(dataset_hparams)
        representation_layer = get_by_model_name(
            model_hparams, PROTO_REPRESENTATION_LAYER, "using last layer for prototypes")

        warnings.warn(f"Generating scores...")
        with stopwatch:
            scores = representation_metrics(
                model, dataloader, device=args.device,
                to_cpu=True, to_numpy=True,
                pd_layers=layers,
                generate_pointwise_metrics=True,
                verbose=args.verbose,
                pd_append_softmax=True,
                pd_train_labels=consensus_labels,
                pd_return_layerpred=True,
                use_faiss=True,
                proto_layer=representation_layer,
                selfproto_k=n_classes,
                selfproto_random_state=args.seed,
            )
        with stopwatch:
            scores["grand"] = grand_score(model, dataloader, device=args.device).detach().cpu()
        # save scores per replicate
        save_scores(save_dir, scores, ep_it)
    print(f"Finished {args.ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--path_to_data_hparam', required=True, type=Path)
    parser.add_argument('--train', default=False, action="store_true")
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--batch_size', default=5000, type=int)
    parser.add_argument('--device', default="cuda", type=str)  # only affects KNN of self-supervised prototypes
    parser.add_argument('--seed', default=42, type=int)  # only affects KNN of self-supervised prototypes

    args = parser.parse_args()
    generate_scores(args)
