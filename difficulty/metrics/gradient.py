from typing import Dict, List
from pathlib import Path
import torch
import torch.nn as nn
from difficulty.metrics.pointwise import softmax, class_confidence
from difficulty.metrics.accumulator import Accumulator, OnlineVariance


__all__ = [
    "ClassOutput",
    "OnlineVarianceOfGradients",
    "input_gradient",
    "input_gradient_from_dataloader",
    "mean_color_channels",
    "mean_pixels",
    "variance_of_gradients",
]


class ClassOutput(nn.Module):
    def __init__(self, softmax=True, use_argmax_labels=False) -> None:
        super().__init__()
        self.softmax = softmax
        self.use_argmax_labels = use_argmax_labels

    def forward(self, x, labels):
        if self.softmax:
            x = softmax(x)
        if self.use_argmax_labels:
            labels = torch.argmax(x, dim=-1).detach()
        conf = class_confidence(x, labels)
        return conf


def _eval_loop(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        inner_fn: callable,
        device: str="cpu",
):  # note: this may require grad, hence is separate from eval.py
    model = model.to(device=device)
    for j, (data, labels) in enumerate(dataloader):
        data = data.to(device=device)
        labels = labels.to(device=device)
        yield inner_fn(model, data, labels)


def input_gradient(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor=None,
        loss_fn: callable=ClassOutput(),
):
    model.eval()
    # prevent input from belonging to multiple computation graphs
    inputs = inputs.detach()
    grad_flags = {k: v.requires_grad for k, v in model.named_parameters()}
    model.requires_grad_(False)  # only need grad for inputs
    inputs.requires_grad_(True)  # in place operation
    y = model(inputs)
    loss = loss_fn(y, labels)
    loss.backward(torch.ones_like(loss))
    gradient = inputs.grad.detach()
    for k, v in model.named_parameters():
        v.requires_grad_(grad_flags[k])  # reset grad flags
    return gradient


def input_gradient_from_dataloader(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: callable=ClassOutput(),
        device: str="cpu",
):
    eval_fn = lambda m, d, l: input_gradient(m, d, l, loss_fn=loss_fn)
    return torch.cat([*_eval_loop(model, dataloader, eval_fn, device)], dim=0)


def mean_color_channels(images: torch.Tensor, channel_dim=-3):
    return torch.mean(images, dim=channel_dim)


def mean_pixels(images: torch.Tensor):
    return torch.mean(images.reshape(images.shape[0], -1), dim=1)


def variance_of_gradients(
        models: List[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        use_predicted_labels=False,
        channel_dim=-3,
        device="cpu",
):
    """Agarwal, C., D'souza, D., & Hooker, S. (2022).
    Estimating example difficulty using variance of gradients.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10368-10378).

    Take V_p = \sqrt{1/K} \sum_{t=1}^K (S_t - \mu_p)^2 over timesteps, then take 1/N \sum_{p=1}^N V_p over pixels

    Args:
        input_gradients (torch.Tensor): pre-softmax activation gradient indexed at predicted/true label with respect to the input
            For example, a 32x32x3 image will have a gradient of the same shape.
            Shape is (N, ...) where N is number of examples and the remainder are image dimensions.
        channel_dim (int): dimension of color channels which is averaged over. Defaults to -3,
            corresponding to images with (C, H, W) shape
    """
    # average gradient over color channels
    grad = []
    for model in models:
        g = input_gradient_from_dataloader(model, dataloader, loss_fn=ClassOutput(
            softmax=True, use_argmax_labels=use_predicted_labels), device=device)
        grad.append(mean_color_channels(g, channel_dim=channel_dim))
    # compute variance over all timesteps
    grad = torch.stack(grad, dim=0)
    vog = torch.var(grad, dim=0)
    # average variance over all pixels
    return mean_pixels(vog)


class OnlineVarianceOfGradients(Accumulator):

    def __init__(self,
                 use_predicted_labels=False,
                 channel_dim=-3,
                 n=None,
                 sum=None,
                 sum_sq=None,
                 dtype: torch.dtype=torch.float64,
                 device: str="cpu",
                 metadata_lists: Dict[str, list]={},
                 **metadata
    ):
        super().__init__(dtype=dtype, device=device, metadata_lists=metadata_lists, use_predicted_labels=use_predicted_labels, channel_dim=channel_dim, **metadata)
        self.var = OnlineVariance(n=n, sum=sum, sum_sq=sum_sq, dtype=dtype, device=device)
        self.loss_fn = ClassOutput(softmax=True, use_argmax_labels=use_predicted_labels)

    def save(self, file: Path):
        super().save(file, n=self.var.mean.n, sum=self.var.mean.sum, sum_sq=self.var.sum_sq)

    def add(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, **metadata):
        super()._add(torch.ones(1), **metadata)
        g = input_gradient_from_dataloader(model, dataloader, loss_fn=self.loss_fn, device=str(self.metadata["device"]))
        grad = mean_color_channels(g, channel_dim=int(self.metadata["channel_dim"]))
        self.var.add(grad, dim=None)
        return self

    def get(self):
        vog = self.var.get()
        return mean_pixels(vog)


#TODO GraNd score from DL on a data diet

#TODO linear approximation of adversarial input margin (Jiang et al., 2018 c.f. Baldock)

