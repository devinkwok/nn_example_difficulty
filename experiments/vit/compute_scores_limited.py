# only computes supervised/self proto, el2n, conf, margin, loss from vit outputs
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append("./src")
from difficulty.metrics import representation, pointwise
from difficulty.utils import detach_tensors
import pandas as pd
import torch
import torchvision
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--models_csv", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--i", type=int, required=True)
parser.add_argument("--debug", type=bool, default=False)
args = parser.parse_args()


data_root = str(Path.home() / "/scratch/data/")
output_dir =  str(Path.home() / "/scratch/2023-difficulty/vit_outputs/")


def get_filename(folder, epoch):
    filename = f'{output_dir}/{args.dataset}/{folder}/output_ep{epoch}_it0.npz'
    return filename


def get_targets(dataset_name):
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(f'{data_root}/cifar10', train=True, download=False)
        labels = dataset.targets
        classes = dataset.classes
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(f'{data_root}/cifar100', train=True, download=False)
        labels = dataset.targets
        classes = dataset.classes
    elif dataset_name == "imagenet":
        dataset = torchvision.datasets.ImageNet(f'{data_root}/imagenet', "train")
        labels = [y for _, y in dataset.samples]
        classes = [", ".join(x) for x in dataset.classes]  # this should be sorted in wordnet id order
    return labels, classes

keep = pd.read_csv(args.models_csv)
row = keep.iloc[args.i]
name = row["full_name"]
save_dir = Path(get_filename(name, 0)).parent / "ckptmetrics"
if save_dir.exists():
    print("Skipping", name)
else:
    labels, classes = get_targets(args.dataset)
    labels = torch.tensor(labels)

    for epoch in tqdm([row["early_epoch"], row["epoch"]]):
        file = get_filename(name, epoch)
        print(file)

        arrays = np.load(file)
        prob = torch.tensor(arrays["prob"])
        representations = torch.tensor(arrays["repr"])

        output_file_suffix = "_".join(Path(file).name.split("_")[1:])
        if args.debug:
            print((save_dir / f"key_{output_file_suffix}"))
        else:
            scores = {
                **pointwise.pointwise_metrics(prob, labels, has_softmax_applied=True),
                **detach_tensors({
                    "proto": representation.supervised_prototypes(representations, labels),
                    "selfproto": representation.self_supervised_prototypes(representations, k=len(classes)),
                }, to_cpu=True, to_numpy=True),
            }
            save_dir.mkdir(exist_ok=True, parents=True)
            for k, v in scores.items():
                np.savez(save_dir / f"{k}_{output_file_suffix}", v)
