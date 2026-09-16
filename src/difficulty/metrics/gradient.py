from collections import defaultdict
from typing import Dict, List, Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
from functorch import make_functional_with_buffers, jacrev
from difficulty.metrics.pointwise import softmax, class_confidence
from difficulty.metrics.accumulator import Accumulator, OnlineVariance
from difficulty.utils import get_dtype, match_key, concat_metrics, detach_tensors


__all__ = [
    "gradient_product_scores",
    "softmax_class_confidence",
    "input_gradient",
    "input_gradient_from_dataloader",
    "mean_color_channels",
    "mean_pixels",
    "variance_of_gradients",
    "OnlineVarianceOfGradients",
    "functional_gradient",
    "gradient_norm",
    "grand_score",
    "get_val_gradient",
    "tracin_score",
]


def gradient_product_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    val_samples: List[int]=None,
    val_seeds: List[int]= None,
    val_loss_fn=None,
    loss_fn=None,
    include: Optional[List[str]]=None,
    exclude: Optional[List[str]]=None,
    device: str="cpu",
    to_cpu=True,
    to_numpy=False,
    dtype: Union[str, torch.dtype]=torch.float64,
):
    """Computes both TracIN (`tracin_score`) and GraNd (`grand_score`).

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        val_dataloader (torch.utils.data.DataLoader): Dataloader containing data to sample validation gradients from for TracIN.
        val_samples (List[int], optional): How many validation examples to use. For each value in the list,
            a separate TracIN score is computed via an independent random sample of val_dataloader.
            If None, uses all examples in val_dataloader. Defaults to None.
        val_seeds (List[int], optional): If set, use these seeds to sample from val_dataloader.dataset.
            Defaults to None.
        val_loss_fn (_type_, optional): loss function for validation gradients, must have reduction="sum".
            If None, uses nn.CrossEntropyLoss. Defaults to None.
        loss_fn (_type_, optional): loss function for gradients, must have reduction="none".
            If None, uses nn.CrossEntropyLoss. Defaults to None.
        include (Optional[List[str]], optional): if set, only use parameters that have
            at least one of these strings in their name. Defaults to None.
        exclude (Optional[List[str]], optional): if set, do not use parameters that have
             any of these strings in their name. Defaults to None.
            Defaults to torch.float64.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        to_cpu (bool, optional): if results should be moved to cpu. Defaults to True.
        to_numpy (bool, optional): if results should be converted to numpy arrays. Defaults to False.
        dtype (Union[str, torch.dtype], optional): Data type to transform gradients in after backward pass.

    Returns:
        Dict[str, torch.Tensor]: dictionary of scores. Keys have `partial` appended
            if computed using gradients filtered to a subset of parameters by include and exclude.
    """
    dtype = get_dtype(dtype)
    if val_samples is None:
        val_samples = [len(val_dataloader.dataset)]
    if val_seeds is None:
        val_seeds = [None] * len(val_samples)

    # get validation gradients
    val_grad = []
    for n, s in zip(val_samples, val_seeds):
        grad = get_val_gradient(model, val_dataloader, device, loss_fn=val_loss_fn, n_samples=n, seed=s, concat=False)
        val_grad.append(torch.cat([v for k, v in grad]))
    val_grad = torch.stack(val_grad, dim=1)

    # check if include and exclude take a proper subset of the gradients
    partial_grad = [(k, v) for k, v in grad if match_key(k, include=include, exclude=exclude)]
    compute_partial = {k for k, v in grad} != {k for k, v in partial_grad}

    val_grad_partial = []
    if compute_partial:
        for n, s in zip(val_samples, val_seeds):
            val_grad_partial.append(torch.cat([v for k, v in partial_grad]))
        val_grad_partial = torch.stack(val_grad_partial, dim=1)

    def compute_gradient_product_scores(model, inputs, labels):
        scores = {}

        gradients, _ = _functional_gradient(model, inputs, labels, loss_fn=loss_fn, dtype=dtype)
        differing_keys = {k for k, v in grad}.symmetric_difference({k for k, v in gradients})
        if len(differing_keys) > 0:
            raise ValueError(f"Train and validation gradients don't match, set requires_grad=False for parameters not affected by loss.backward(). Affected keys: {differing_keys}")
        scores["grand"] = _filter_and_transform_gradient(gradients, grad_transform=lambda x: torch.linalg.norm(x, dim=-1))

        val_dotprod = _filter_and_transform_gradient(gradients, grad_transform=lambda x: x @ val_grad)
        scores.update({f"tracin{n}": val_dotprod[:, i] for i, n in enumerate(val_samples)})

        if compute_partial:
            scores["grandpartial"] = _filter_and_transform_gradient(
                gradients, grad_transform=lambda x: torch.linalg.norm(x, dim=-1), include=include, exclude=exclude)

            val_dotprod_partial = _filter_and_transform_gradient(
                gradients, grad_transform=lambda x: x @ val_grad_partial, include=include, exclude=exclude)
            scores.update({f"tracin{n}partial": val_dotprod_partial[:, i] for i, n in enumerate(val_samples)})

        return scores

    # collect each batch's dict of results together
    scores = defaultdict(list)
    # for score_dict in _eval_loop(model, dataloader, eval_fn, device):
    for score_dict in _eval_loop(model, dataloader, compute_gradient_product_scores, device):
        for k, v in score_dict.items():
            scores[k].append(v)

    scores = concat_metrics(scores)
    scores = detach_tensors(scores, to_cpu=to_cpu, to_numpy=to_numpy)
    return scores


def softmax_class_confidence(x, labels):
    return class_confidence(softmax(x), labels)


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
        loss_fn: callable=softmax_class_confidence,
        return_output: bool=False,
        use_argmax_labels: bool=False,
        dtype: Union[str, torch.dtype]=torch.float64,
):
    dtype = get_dtype(dtype)
    model.eval()
    # prevent input from belonging to multiple computation graphs
    inputs = inputs.detach()
    grad_flags = {k: v.requires_grad for k, v in model.named_parameters()}
    model.requires_grad_(False)  # only need grad for inputs
    inputs.requires_grad_(True)  # in place operation
    y = model(inputs)
    if use_argmax_labels:
        labels = torch.argmax(y, dim=-1)
    loss = loss_fn(y, labels)
    loss.backward(torch.ones_like(loss))
    gradient = inputs.grad.detach().to(dtype=dtype)
    for k, v in model.named_parameters():
        v.requires_grad_(grad_flags[k])  # reset grad flags
    if return_output:
        return gradient, y.detach().to(dtype=dtype)
    return gradient


def input_gradient_from_dataloader(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: callable=softmax_class_confidence,
        device: str="cpu",
        return_output: bool=False,
        use_argmax_labels: bool=False,
        dtype: Union[str, torch.dtype]=torch.float64,
):
    eval_fn = lambda m, d, l: input_gradient(
        m, d, l, loss_fn=loss_fn, return_output=True, use_argmax_labels=use_argmax_labels, dtype=dtype)
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
        loss_fn: callable=softmax_class_confidence,
        use_predicted_labels=False,
        channel_dim=-3,
        dtype: Union[str, torch.dtype]=torch.float64,
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
        grad = input_gradient_from_dataloader(
            model, dataloader, loss_fn=loss_fn, device=device,
            return_output=False, use_argmax_labels=use_predicted_labels, dtype=dtype)
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
                 dtype: Union[str, torch.dtype]=torch.float64,
                 loss_fn: callable=softmax_class_confidence,
                 use_argmax_labels: bool=False,
                 device: str="cpu",
                 metadata_lists: Dict[str, list]={},
                 **metadata
    ):
        super().__init__(dtype=dtype, device=device, metadata_lists=metadata_lists, use_predicted_labels=use_predicted_labels, channel_dim=channel_dim, **metadata)
        self.var = OnlineVariance(n=n, sum=sum, sum_sq=sum_sq, dtype=dtype, device=device)
        self.use_argmax_labels = use_argmax_labels
        self.loss_fn = loss_fn

    def save(self, file: Path):
        super().save(file, n=self.var.mean.n, sum=self.var.mean.sum, sum_sq=self.var.sum_sq)

    def add(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, return_output: bool=False, **metadata):
        super()._add(torch.ones(1), **metadata)
        grad = input_gradient_from_dataloader(
            model, dataloader, loss_fn=self.loss_fn, device=str(self.get_metadata("device")),
            return_output=return_output, use_argmax_labels=self.use_argmax_labels, dtype=self.get_metadata("dtype"))
        if return_output:
            grad, out = grad
        gradients = mean_color_channels(grad, channel_dim=int(self.get_metadata("channel_dim")))
        self.var.add(gradients, dim=None)
        if return_output:
            return self, out
        return self

    def get(self):
        vog = self.var.get()
        return mean_pixels(vog)

    def get_mean(self):
        return self.var.get_mean()


def _functional_gradient(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn=None,
        dtype: Union[str, torch.dtype]=torch.float64,
):
    dtype = get_dtype(dtype)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    def model_loss(p, x, z):
        y = torch.func.functional_call(model, p, x)
        loss = loss_fn(y, z)
        return loss, y

    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if params == {}:
        raise ValueError(f"No parameters require grad in model: {[(k, v.requires_grad) for k, v in model.named_parameters()]}")
    jacobian, outputs = jacrev(model_loss, argnums=0, has_aux=True)(params, inputs, labels)
    n_examples = inputs.shape[0]
    gradients = [(k, v.detach().reshape(n_examples, -1).to(dtype=dtype)) for k, v in jacobian.items()]
    return gradients, outputs.detach().to(dtype=dtype)


def _filter_and_transform_gradient(gradients, grad_transform=None, include=None, exclude=None):
    gradients = [v for k, v in gradients if match_key(k, include=include, exclude=exclude)]
    gradients = torch.cat(gradients, dim=-1)
    if grad_transform is not None:
        gradients = grad_transform(gradients)
    return gradients


def functional_gradient(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        grad_transform=None,
        loss_fn=None,
        return_output: bool=False,
        dtype: Union[str, torch.dtype]=torch.float64,
        include: Optional[List[str]]=None,
        exclude: Optional[List[str]]=None,
):
    gradients, outputs = _functional_gradient(model, inputs, labels, loss_fn=loss_fn, dtype=dtype)
    gradients = _filter_and_transform_gradient(gradients, grad_transform=grad_transform, include=include, exclude=exclude)
    if return_output:
        return gradients, outputs
    return gradients


def param_gradient(
        model: nn.Module,
        include: Optional[List[str]]=None,
        exclude: Optional[List[str]]=None,
        concat: bool=True,
        scale: float=1,
        dtype: Union[str, torch.dtype]=torch.float64
):
    dtype = get_dtype(dtype)
    grad = []
    for name, param in model.named_parameters():
        if hasattr(param, "grad") and param.grad is not None and match_key(name, include=include, exclude=exclude):
            grad.append((name, param.grad.detach().flatten().to(dtype=dtype) * scale))
    if concat:
        grad = torch.cat([v for k, v in grad])
    return grad


def _gradient(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn=None,
        include: Optional[List[str]]=None,
        exclude: Optional[List[str]]=None,
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    grad_flags = {k: v.requires_grad for k, v in model.named_parameters()}
    model.requires_grad_(True)  # only need grad for inputs
    gradients, outputs = [], []
    for x, z in zip(inputs, labels):
        model.zero_grad()
        y = model(x.unsqueeze(0))
        loss = loss_fn(y, z.unsqueeze(0))
        loss.backward()
        gradients.append(param_gradient(model, include=include, exclude=exclude))
        outputs.append(y.detach())
    for k, v in model.named_parameters():
        v.requires_grad_(grad_flags[k])  # reset grad flags
    return torch.stack(gradients, dim=0), torch.stack(outputs, dim=0)


def gradient_norm(
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: callable=nn.CrossEntropyLoss(reduction="none"),
        return_output: bool=False,
        dtype: Union[str, torch.dtype]=torch.float64,
        include: Optional[List[str]]=None,
        exclude: Optional[List[str]]=None,
):
    dtype = get_dtype(dtype)
    gradients, outputs = _gradient(model, inputs, labels=labels, loss_fn=loss_fn, include=include, exclude=exclude)
    grad_norm = torch.linalg.norm(gradients, dim=1).to(dtype=dtype)
    if return_output:
        return grad_norm, outputs.to(dtype=dtype)
    return grad_norm


def grand_score(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: callable=nn.CrossEntropyLoss(reduction="none"),
        device: str="cpu",
        use_functional: bool=True,
        return_output: bool=False,
        dtype: Union[str, torch.dtype]=torch.float64,
        include: Optional[List[str]]=None,
        exclude: Optional[List[str]]=None,
):
    """Paul, M., Ganguli, S., & Dziugaite, G. K. (2021).
    Deep learning on a data diet: Finding important examples early in training.
    Advances in Neural Information Processing Systems, 34, 20596-20607.

    GraNd score: norm of flattened per-example gradient.

        return_output (bool, optional): Return outputs, i.e. model(data),
            to avoid having to call eval twice when generating other metrics. Defaults to False.
    """
    if use_functional:
        eval_fn = lambda m, d, l: functional_gradient(
            m, d, l, loss_fn=loss_fn, grad_transform=lambda x: torch.linalg.vector_norm(x, dim=-1),
            return_output=return_output, dtype=dtype, include=include, exclude=exclude,
        )
    else:
        eval_fn = lambda m, d, l: gradient_norm(
            m, d, l, loss_fn=loss_fn, return_output=return_output,
            dtype=dtype, include=include, exclude=exclude
        )
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


def _get_random_samples(dataloader, n_samples, seed=None):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataloader.dataset), generator=generator)[:n_samples]
    subset = torch.utils.data.Subset(dataloader.dataset, idx)
    subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=dataloader.batch_size)
    return subset_dataloader


def get_val_gradient(
    model,
    dataloader,
    device: str="cpu",
    loss_fn=None,
    n_samples: Optional[int]=None,
    seed=None,
    include: Optional[List[str]]=None,
    exclude: Optional[List[str]]=None,
    concat: bool=True,
    dtype: Union[str, torch.dtype]=torch.float64,
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
    if n_samples is not None and n_samples < len(dataloader.dataset):
        dataloader = _get_random_samples(dataloader, n_samples, seed=seed)

    model = model.to(device=device)
    model.eval()  # freeze batchnorm

    model.zero_grad()
    for x, y in dataloader:
        y_pred = model(x.to(device=device))
        loss = loss_fn(y_pred, y.to(device=device))
        loss.backward()

    # divide by number of samples to get mean gradient
    n = len(dataloader.dataset)
    val_grad = param_gradient(model, include=include, exclude=exclude, concat=concat, scale=1 / n, dtype=dtype)
    return val_grad


def tracin_score(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    val_gradient: torch.Tensor,
    loss_fn=None,
    device: str="cpu",
    dtype: Union[str, torch.dtype]=torch.float64,
    include: Optional[List[str]]=None,
    exclude: Optional[List[str]]=None,
):
    """
    Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020).
    Estimating training data influence by tracing gradient descent.
    Advances in Neural Information Processing Systems, 33, 19920-19930.

    TracIN: dot product of per-example gradient with gradient of one or more validation examples.

        val_gradient (torch.Tensor): gradient in the shape (V, P) where P is number of parameters,
            and V is an arbitrary number of vectors to compute dot products over.
            Can fill this using `get_val_gradient`.
    """
    if len(val_gradient.shape) == 1:
        val_gradient = val_gradient.reshape(1, -1)
    eval_fn = lambda m, d, l: functional_gradient(
        m, d, l, loss_fn=loss_fn, grad_transform=lambda x: x @ val_gradient.T,
        return_output=False, dtype=dtype, include=include, exclude=exclude,
    )
    scores = list(_eval_loop(model, dataloader, eval_fn, device))
    scores = torch.cat(scores, dim=0)  # (N, V) where N is number of examples
    return scores


#TODO linear approximation of adversarial input margin (Jiang et al., 2018 c.f. Baldock)
