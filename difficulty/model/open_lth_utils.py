import json
import os
from pathlib import Path
from typing import Dict
import torch
import torchvision

import sys
from open_lth.foundations.step import Step
sys.path.append("open_lth")
from open_lth.models import registry
from open_lth.foundations.hparams import DatasetHparams, ModelHparams
from open_lth.datasets import registry as datasets_registry


def get_hparams(save_dir: Path) -> Dict[str, str]:
    # find first available hparam file
    hparam_file = None
    for root, _, files in os.walk(save_dir):
        for file in files:
            if file == "hparams.log":
                hparam_file = Path(root) / Path(file)
                break
    if hparam_file is None:
        raise RuntimeError(f"hparam file not found in {save_dir}")
    with open(hparam_file) as f:
        hparam_lines = f.readlines()
    # parse text into dict
    hparams = {}
    for line in hparam_lines:
        line = line.strip()
        if line.endswith(" Hyperparameters"):
            header = line[:-len(" Hyperparameters")]
            hparams[header] = {}
        elif line.startswith("* "):
            k, v = line[len("* "):].split(" => ")
            hparams[header][k] = v
        else:
            raise ValueError(line)
    return hparams


def get_model(hparams: Dict[str, str], device: str) -> torch.nn.Module:
    model = registry.get(ModelHparams(
        hparams["Model"]["model_name"],
        hparams["Model"]["model_init"],
        hparams["Model"]["batchnorm_init"])).to(device=device)
    return model


def get_dataset(hparams, train, data_root):
    if hparams["Dataset"]["dataset_name"] == "cifar10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.CIFAR10(root=data_root / "cifar10",
                    train=train, download=False, transform=transforms)
    elif hparams["Dataset"]["dataset_name"] == "mnist":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = torchvision.datasets.MNIST(root=data_root / "mnist",
                    train=train, download=False, transform=transforms)
    else:
        raise ValueError(f"Unsupported dataset {hparams['Dataset']['dataset_name']}")
    return dataset


def get_iterations_per_epoch(hparams):
    dataset_hparams = DatasetHparams(
        dataset_name=hparams["Dataset"]["dataset_name"],
        batch_size=int(hparams["Dataset"]["batch_size"]),
        )
    return datasets_registry.iterations_per_epoch(dataset_hparams)


def get_replicate_dir(ckpt_dir: Path, replicate: int):
    return ckpt_dir / f"replicate_{replicate}"


def get_training_checkpoints(hparams, ckpt_dir: Path, replicate: int):
    iterations_per_epoch = get_iterations_per_epoch(hparams)
    subdir = get_replicate_dir(ckpt_dir, replicate) / "level_pretrain" / "main"
    if not subdir.exists():
        raise ValueError(f"Training checkpoint dir not available: {subdir}")
    ckpts = []
    for file in subdir.glob("*"):
        try:
            _, ep, it = file.stem.split("_")
            ep = int(ep.split("ep")[1])
            it = int(it.split("it")[1])
            step = Step.from_epoch(ep, it, iterations_per_epoch)
            ckpts.append((step, file))
        except ValueError:  # not a valid checkpoint
            print(f"{file.name} is not a checkpoint")
    sorted(ckpts, key=lambda x: x[0])  # order by step
    return list(zip(*ckpts))  # return as separate lists (steps, filenames)


def get_latest_training_checkpoint(hparams, ckpt_dir: Path, replicate: int):
    steps, ckpts = get_training_checkpoints(hparams, ckpt_dir, replicate)
    return steps[-1], ckpts[-1]


def get_pruning_sparsity(pruning_dir: Path):
    sparsity_file = pruning_dir / "main" / "sparsity_report.json"
    if not sparsity_file.exists():
        raise RuntimeError(f"Pruning sparsity file missing: {sparsity_file}")
    with open(sparsity_file, 'r') as f:
        sparsity_dict = json.loads(f.read())
    return sparsity_dict["unpruned"] / sparsity_dict["total"]


def get_iterative_magnitude_pruning_checkpoints(hparams, ckpt_dir: Path, replicate: int):
    pruning_fraction = float(hparams["Pruning"]["pruning_fraction"])
    replicate_dir = get_replicate_dir(ckpt_dir, replicate)
    # last_ckpt has same name as pruning ckpts
    _, last_ckpt = get_latest_training_checkpoint(hparams, ckpt_dir, replicate)
    ckpts = []
    for subdir in replicate_dir.glob("*"):
        if subdir.is_dir() and subdir.name.startswith("level_"):
            try:
                prune_level = int(subdir.name.split("_")[1])
            except ValueError:  # not a valid checkpoint
                print(f"{subdir} does not contain a pruning checkpoint")
                continue
            ckpt_file = subdir / "main" / last_ckpt.name
            if not ckpt_file.exists():
                print(f"{ckpt_file} is not a pruning checkpoint")
                continue
            # check that fractions are close
            fraction = (1 - pruning_fraction)**prune_level
            actual_fraction = get_pruning_sparsity(subdir)
            if abs(fraction - actual_fraction) > 1e-4:
                print(f"WARNING: pruning fractions differ: calculated {fraction}, actual {actual_fraction}")
            ckpts.append((actual_fraction, ckpt_file))
    sorted(ckpts, key=lambda x: x[0])  # order by step
    return list(zip(*ckpts))  # return as separate lists (steps, filenames)
