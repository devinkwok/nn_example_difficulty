from pathlib import Path
import traceback
from tqdm import tqdm
import warnings
import torch
import pandas as pd
import numpy as np
from PIL import Image
import open_clip
import torchvision
from huggingface_hub import hf_hub_download
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--models_csv", type=str, required=True)
parser.add_argument("--i", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=1024)
args = parser.parse_args()


repo_id = "laion/scaling-laws-for-comparison"
repo_type = "model"
local_dir = str(Path.home() / "/scratch/2023-difficulty/vit_ckpts/")
output_dir = str(Path.home() / "/scratch/2023-difficulty/vit_outputs/")
data_root = "/tmp/data/"


def get_filename(folder, epoch, is_local=False, output_name=None):
    filename = f'{folder}/epoch_{epoch}.pt'
    if output_name is not None:
        filename = f'{output_dir}/{args.dataset}/{folder}/{output_name}_ep{epoch}_it0.npz'
        Path()
    elif is_local:
        filename = f'{local_dir}/{filename}'
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    return filename


def get_dataloader_and_labels(dataset_name, transform):
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(f'{data_root}/cifar10', train=True, download=False, transform=transform)
        classes = dataset.classes
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(f'{data_root}/cifar100', train=True, download=False, transform=transform)
        classes = dataset.classes
    elif dataset_name == "imagenet":
        dataset = torchvision.datasets.ImageNet(f'{data_root}/imagenet', "train", transform=transform)
        classes = [", ".join(x) for x in dataset.classes]  # this should be sorted in wordnet id order
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=(10 if args.debug else args.batch_size), shuffle=False, num_workers=6)
    return dataloader, classes


keep = pd.read_csv(args.models_csv)
row = keep.iloc[args.i]

for epoch in [row["early_epoch"], row["epoch"]]:
    name = row["full_name"]
    output_file = get_filename(name, epoch, output_name="output")
    if Path(output_file).exists():
        print(f"Skipping {output_file}")
        continue
    try:
        with torch.no_grad(), torch.amp.autocast('cuda'):
            local_path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=get_filename(name, epoch), local_dir=local_dir)

            model_type, model_size = local_path.split("/")[-2].split("_")[:2]
            if model_type == "mammut":
                model_id = f'{model_type}_{model_size}'
            else:
                model_id = model_size

            model, _, transform = open_clip.create_model_and_transforms(model_id, pretrained=local_path)
            model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
            model = model.cuda()
            tokenizer = open_clip.get_tokenizer(model_id)
            dataloader, classes = get_dataloader_and_labels(args.dataset, transform)

            # precompute the label tokens
            text = tokenizer(classes).cuda()
            text_features = model.encode_text(text.cuda())
            text_features /= text_features.norm(dim=-1, keepdim=True)

            all_features = []
            all_probs = []
            for x, y in tqdm(dataloader):
                x = x.cuda()
                y = y.cuda()
                image_features = model.encode_image(x)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                all_features.append(image_features.detach().cpu().numpy())
                all_probs.append(text_probs.detach().cpu().numpy())
                acc = (torch.sum(torch.argmax(text_probs, dim=1) == y) / len(y)).detach().cpu().item()

                print(name, epoch, acc, "Label probs:", text_probs.shape)  # prints: [[1., 0., 0.]]
                if args.debug:
                    break
            # save representations and probabilities
            all_probs = np.concatenate(all_probs, axis=0) 
            all_features = np.concatenate(all_features, axis=0) 
            np.savez(output_file, prob=all_probs, repr=all_features)

    except Exception as e:
        warnings.warn(f'FAILED: {name} {epoch}\n\t{str(e)} {traceback.format_exc()}')
        failed = pd.DataFrame({"full_name": name, "epoch": epoch, "error": str(e), "traceback": traceback.format_exc()})
        failed.to_csv(f"failed-{args.i}.csv")

    if args.debug:
        break
