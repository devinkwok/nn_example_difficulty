from typing import Dict, List
from pathlib import Path
import torch
import torch.nn as nn
from functorch import jacrev, make_functional_with_buffers
from difficulty.metrics.pointwise import softmax, class_confidence
from difficulty.metrics.accumulator import Accumulator, OnlineVariance


__all__ = [
    "ClassOutput",
    "input_gradient",
    "input_gradient_from_dataloader",
    "mean_color_channels",
    "mean_pixels",
    "variance_of_gradients",
    "OnlineVarianceOfGradients",
    "gradient_norm",
    "functional_gradient_norm",
    "grand_score",
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
        return_output: bool=False,
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
    if return_output:
        return gradient, y.detach()
    return gradient


def input_gradient_from_dataloader(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: callable=ClassOutput(),
        device: str="cpu",
        return_output: bool=False,
):
    eval_fn = lambda m, d, l: input_gradient(m, d, l, loss_fn=loss_fn, return_output=True)
    iterator = _eval_loop(model, dataloader, eval_fn, device)
    gradients, outputs = [], []
    for gradient, output in iterator:
        gradients.append(gradient)
        if return_output:
            outputs.append(output)
    gradients = torch.cat(gradients, dim=0)
    if return_output:
        return gradients, torch.cat(outputs, dim=0)
    return gradients


def mean_color_channels(images: torch.Tensor, channel_dim=-3):
    return torch.mean(images, dim=channel_dim)


def mean_pixels(images: torch.Tensor):
    return torch.mean(images.reshape(images.shape[0], -1), dim=1)


def variance_of_gradients(
        models: List[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        device="cpu",
        use_predicted_labels=False,
        channel_dim=-3,
):
    """Agarwal, C., D'souza, D., & Hooker, S. (2022).
    Estimating example difficulty using variance of gradients.
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10368-10378).

    Take V_p = \sqrt{1/K} \sum_{t=1}^K (S_t - \mu_p)^2 over timesteps, then take 1/N \sum_{p=1}^N V_p over pixels

    Args:
        channel_dim (int): dimension of color channels which is averaged over. Defaults to -3,
            corresponding to images with (C, H, W) shape
    """
    # average gradient over color channels
    gradients = []
    for model in models:
        grad = input_gradient_from_dataloader(model, dataloader, loss_fn=ClassOutput(
            softmax=True, use_argmax_labels=use_predicted_labels), device=device, return_output=False)
        gradients.append(mean_color_channels(grad, channel_dim=channel_dim))
    # compute variance over all timesteps
    gradients = torch.stack(gradients, dim=0)
    # average variance over all pixels
    vog = mean_pixels(torch.var(gradients, dim=0))
    return vog


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

    def add(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, return_output: bool=False, **metadata):
        """
        Args:
            model (nn.Module): _description_
            dataloader (torch.utils.data.DataLoader): _description_
            return_output (bool, optional): Return outputs, i.e. model(data),
                to avoid having to call eval twice when generating other metrics. Defaults to False.

        Returns:
            _type_: _description_
        """
        super()._add(torch.ones(1), **metadata)
        grad = input_gradient_from_dataloader(
            model, dataloader,
            loss_fn=self.loss_fn, device=str(self.metadata["device"]),
            return_output=return_output)
        if return_output:
            grad, out = grad
        gradients = mean_color_channels(grad, channel_dim=int(self.metadata["channel_dim"]))
        self.var.add(gradients, dim=None)
        if return_output:
            return self, out
        return self

    def get(self):
        vog = self.var.get()
        return mean_pixels(vog)


def functional_gradient_norm(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor=None,
        loss_fn: callable=nn.CrossEntropyLoss(reduction="none"),
        return_output: bool=False,
):
    model.eval()
    func_model, params, buffers = make_functional_with_buffers(model)

    def model_loss(p, b, x, z):
        y = func_model(p, b, x)
        loss = loss_fn(y, z)
        if return_output:
            return loss, y
        return loss

    n_examples = inputs.shape[0]
    jacobian = jacrev(model_loss, argnums=0, has_aux=return_output)(params, buffers, inputs, labels)
    if return_output:
        jacobian, outputs = jacobian
    gradients = [x.detach().reshape(n_examples, -1) for x in jacobian]
    gradients = torch.cat(gradients, dim=-1)
    grad_norm = torch.linalg.vector_norm(gradients, dim=-1)
    if return_output:
        return grad_norm, outputs.detach()
    return grad_norm


def gradient_norm(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor=None,
        loss_fn: callable=nn.CrossEntropyLoss(reduction="none"),
        return_output: bool=False,
):
    model.eval()
    grad_flags = {k: v.requires_grad for k, v in model.named_parameters()}
    model.requires_grad_(True)  # only need grad for inputs
    grad_norm, outputs = [], []
    for x, z in zip(inputs, labels):
        model.zero_grad()
        y = model(x.unsqueeze(0))
        loss = loss_fn(y, z.unsqueeze(0))
        loss.backward()
        gradient = torch.cat([x.grad.detach().flatten() for x in model.parameters()])
        grad_norm.append(torch.linalg.norm(gradient).item())
        if return_output:
            outputs.append(y.detach())
    for k, v in model.named_parameters():
        v.requires_grad_(grad_flags[k])  # reset grad flags
    grad_norm = torch.tensor(grad_norm)
    if return_output:
        return grad_norm, torch.cat(outputs, dim=0)
    return grad_norm


def grand_score(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: callable=nn.CrossEntropyLoss(reduction="none"),
        device: str="cpu",
        use_functional: bool=True,
        return_output: bool=False,
):
    """Paul, M., Ganguli, S., & Dziugaite, G. K. (2021).
    Deep learning on a data diet: Finding important examples early in training.
    Advances in Neural Information Processing Systems, 34, 20596-20607.

    GraNd score: norm of flattened per-example gradient.

            return_output (bool, optional): Return outputs, i.e. model(data),
                to avoid having to call eval twice when generating other metrics. Defaults to False.
    """
    if use_functional:
        eval_fn = lambda m, d, l: functional_gradient_norm(
            m, d, l, loss_fn=loss_fn, return_output=return_output)
    else:
        eval_fn = lambda m, d, l: gradient_norm(
            m, d, l, loss_fn=loss_fn, return_output=return_output)
    scores, outputs = [], []
    for grand in _eval_loop(model, dataloader, eval_fn, device):
        if return_output:
            grand, output = grand
            outputs.append(output)
        scores.append(grand)
    scores = torch.cat(scores, dim=0)
    if return_output:
        return scores, torch.cat(outputs, dim=0)
    return scores


#TODO linear approximation of adversarial input margin (Jiang et al., 2018 c.f. Baldock)
