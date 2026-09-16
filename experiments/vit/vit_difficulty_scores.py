from pathlib import Path
import traceback
from tqdm import tqdm
import warnings
import torch
import pandas as pd
import numpy as np
import open_clip
import torchvision
from huggingface_hub import hf_hub_download
import argparse

import sys
sys.path.append("./src")
from difficulty.metrics import representation_metrics, gradient_product_scores
from difficulty.metrics import load_imagenet_classes
from difficulty.model.eval import find_intermediate_layers, evaluate_model


# for prediction depth, get the inputs/outputs to each residual block (post-summation)
PD_INCLUDE_LAYERS = [
    "vit.visual.ln_pre.out",
    "vit.visual.transformer.resblocks.",
]  # note: for softmax, set pd_append_softmax=True in representation_metrics
PD_EXCLUDE_LAYERS = [
    ".ln_1.",
    ".ln_2.",
    ".ls_",
    ".attn.",
    ".mlp.",
]  # this excludes all but the outputs of the residual blocks


def get_dataloader_and_labels(args, transform):
    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            f"{args.data_root}/cifar10", train=True, download=False, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            f"{args.data_root}/cifar10", train=False, download=False, transform=transform
        )
        classes = dataset.classes
    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            f"{args.data_root}/cifar100", train=True, download=False, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            f"{args.data_root}/cifar100", train=False, download=False, transform=transform
        )
        classes = dataset.classes
    elif args.dataset == "imagenet":
        dataset = torchvision.datasets.ImageNet(
            f"{args.data_root}/imagenet", "train", transform=transform
        )
        val_dataset = torchvision.datasets.ImageNet(
            f"{args.data_root}/imagenet", "val", transform=transform
        )
        classes = [", ".join(x) for x in dataset.classes]
        # check that classes are sorted in wordnet id order
        assert np.all(classes == load_imagenet_classes())

    batch_size = args.batch_size
    if args.debug:
        dataset = torch.utils.data.Subset(dataset, torch.arange(99))
        val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(20))
        print(f"DEBUG: {len(dataset)} samples")
        batch_size = 3

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return dataloader, val_dataloader, classes


def get_ckpt(repo_id, repo_type, filename, local_dir):
    local_path = hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=local_dir
    )

    model_type, model_size = local_path.split("/")[-2].split("_")[:2]
    if model_type == "mammut":
        model_id = f"{model_type}_{model_size}"
    else:
        model_id = model_size

    model, _, transform = open_clip.create_model_and_transforms(
        model_id, pretrained=local_path, load_weights_only=False,
    )
    tokenizer = open_clip.get_tokenizer(model_id)

    return model, tokenizer, transform


class VitClassifier(torch.nn.Module):
    def __init__(self, model, tokenizer, classes, device):
        super().__init__()
        self.device = device
        self.vit = model.to(device=self.device)

        # precompute the label tokens, and prevent gradient from going through
        with torch.no_grad():
            text = tokenizer(classes).to(device=self.device)
            text_features = self.vit.encode_text(text.to(device=self.device))
            self.text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).detach().clone()

        # disable grad on text model to prevent functorch from considering them
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.visual.parameters():
            param.requires_grad = True

    def forward(self, x):
        image_features = self.vit.encode_image(x.to(device=self.device))
        norm = image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * (image_features / norm) @ self.text_features.T
        return logits


def get_scores(
    model, dataloader, val_dataloader, val_samples, n_classes, args
):
    torch.autograd.set_detect_anomaly(True, check_nan=False)
    grad_scores = gradient_product_scores(
        model,
        dataloader,
        val_dataloader,
        val_samples=val_samples,
        include=["visual.ln_post.weight"],
        device=args.device,
    )

    outputs, _, _, _ = evaluate_model(model, dataloader, device=args.device)
    predicted_labels = torch.argmax(outputs.detach(), dim=-1)
    pd_layers = find_intermediate_layers(
        model,
        next(iter(dataloader))[0].shape[1:],
        device=args.device,
        include=PD_INCLUDE_LAYERS,
        exclude=PD_EXCLUDE_LAYERS,
        n_test_points=5,
    )
    print("Prediction depth uses layers:", pd_layers)

    rep_scores = representation_metrics(
        model,
        dataloader,
        device=args.device,
        to_cpu=True,
        to_numpy=True,
        pd_layers=pd_layers,
        generate_pointwise_metrics=True,
        verbose=args.verbose,
        pd_append_softmax=True,
        pd_train_labels=predicted_labels,
        pd_return_layerpred=True,
        # use_faiss=True,
        proto_layer="vit.visual.out",
        selfproto_k=n_classes,
    )

    return {
        "pdlayers": pd_layers,
        **rep_scores,
        **grad_scores,
    }


def score_file(directory, score_name, ep_it):
    return directory / f"{score_name}_{ep_it}.npz"


def save_scores(directory, scores, ep_it):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {directory}...")
    for score_name, score in scores.items():
        print(f"... saving {score_name}")
        np.savez(score_file(directory, score_name, ep_it), score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--models_csv", type=str, required=True)
    parser.add_argument("--i", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--repo_id", type=str, default="laion/scaling-laws-for-comparison"
    )
    parser.add_argument("--repo_type", type=str, default="model")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--val_samples", default="1,100,10000", type=str)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.models_csv)
    row = df.iloc[args.i]
    name = row["full_name"]
    save_dir = Path(f"{args.output_root}/{args.dataset}/{name}/")

    for epoch in [row["early_epoch"], row["epoch"]]:
        if (save_dir / f"pdlayers_ep{epoch}_it0.npz").exists():
            print(f"Skipping {name} ep{epoch}")
            continue

        ckpt_name = f"{name}/epoch_{epoch}.pt"
        vit, tokenizer, transform = get_ckpt(
            args.repo_id, args.repo_type, ckpt_name, args.ckpt_root
        )
        dataloader, val_dataloader, classes = get_dataloader_and_labels(args, transform)
        model = VitClassifier(vit, tokenizer, classes, device=args.device)
        print(f"{name}, ep{epoch}, parameters:{torch.sum(torch.tensor([v.numel() for v in model.parameters()]))}")

        val_samples = [int(x) for x in args.val_samples.split(",")]
        if args.debug:
            val_samples = [1, 10]

        scores = get_scores(
            model,
            dataloader,
            val_dataloader,
            val_samples,
            min(len(classes), len(dataloader.dataset)),
            args,
        )
        save_scores(save_dir, scores, f"ep{epoch}_it0")

        if args.debug:
            break


if __name__ == "__main__":
    main()
