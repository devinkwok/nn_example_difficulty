import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import src.resnet50 as resnet_models
sys.path.append(os.path.join(os.environ['HOME'], "lib", "open_lth"))
sys.path.append(os.path.join(os.environ['HOME'], "lib", "nn_example_difficulty"))
import api
from foundations.hparams import DatasetHparams
from difficulty.metrics.representation import self_supervised_prototypes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save_file", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset_hparams = DatasetHparams(dataset_name=args.dataset, batch_size=args.batch_size)
    dataloader = api.get_dataloader(dataset_hparams, train=True, batch_size=args.batch_size)
    model = resnet_models.__dict__["resnet50"](output_dim=0, eval_mode=True)
    state_dict = torch.load("swav_800ep_pretrain.pth.tar")
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    intermediates = []
    for i, (x, labels) in enumerate(dataloader):
        x = x.to(device="cuda")
        y = model(x)
        intermediates.append(y.detach().cpu().numpy())
        print(i, x.shape, y.shape, labels)

    intermediates = np.concatenate(intermediates, axis=0)

    # generate self supervised prototype scores
    selfproto = self_supervised_prototypes(torch.tensor(intermediates), k=args.k, random_state=args.seed)
    args.save_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_file, selfproto.detach().cpu().numpy())
