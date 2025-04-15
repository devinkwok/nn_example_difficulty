"""Following functions assume the last dimension of eval_logits or eval_softmax
is over $C$ classes, and second last dimension is over $N$ examples.
"""
from typing import Union, Dict
import torch
import torch.nn.functional as F

from difficulty.utils import get_dtype, detach_tensors
from difficulty.metrics.accumulator import OnlineVariance


def pointwise_metrics(eval_logits: torch.Tensor,
                      labels: torch.Tensor,
                      to_cpu=True,
                      to_numpy=False,
                      dtype: Union[str, torch.dtype]=torch.float64,
    ) -> Dict[str, torch.Tensor]:
    eval_logits = eval_logits.to(dtype=get_dtype(dtype))
    prob = softmax(eval_logits)
    return detach_tensors({
        "acc": zero_one_accuracy(eval_logits, labels),
        "ent": entropy(prob),
        "conf": class_confidence(prob, labels),
        "maxconf": max_confidence(prob),
        "margin": margin(prob, labels),
        "el2n": error_l2_norm(prob, labels),
    }, to_cpu=to_cpu, to_numpy=to_numpy)


def create_online_pointwise_metrics(dtype=torch.float64, device="cpu", **metadata):
    return {
        "acc": OnlineVariance(dtype=dtype, device=device, **metadata),
        "ent": OnlineVariance(dtype=dtype, device=device, **metadata),
        "conf": OnlineVariance(dtype=dtype, device=device, **metadata),
        "maxconf": OnlineVariance(dtype=dtype, device=device, **metadata),
        "margin": OnlineVariance(dtype=dtype, device=device, **metadata),
        "el2n": OnlineVariance(dtype=dtype, device=device, **metadata),
    }


def softmax(eval_logits: torch.Tensor) -> torch.Tensor:
    softmax = F.softmax(eval_logits, dim=-1)
    return softmax


def zero_one_accuracy(eval_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    acc = torch.argmax(eval_logits, dim=-1) == labels
    return acc


def entropy(eval_softmax: torch.Tensor) -> torch.Tensor:
    # in Shannon entropy, 0 * log(0) = 0
    ent_log =  torch.log(eval_softmax).nan_to_num(nan=0.)
    return -1 * torch.sum(eval_softmax * ent_log, dim=-1)


def class_confidence(eval_softmax: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = torch.unsqueeze(torch.broadcast_to(labels, eval_softmax.shape[:-1]), -1)
    class_conf = torch.take_along_dim(eval_softmax, labels, dim=-1).squeeze(-1)
    return class_conf


def max_confidence(eval_softmax: torch.Tensor) -> torch.Tensor:
    max_conf = torch.max(eval_softmax, dim=-1).values
    return max_conf


def margin(eval_softmax: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """From
    Pleiss, G., Zhang, T., Elenberg, E., & Weinberger, K. Q. (2020).
    Identifying mislabeled data using the area under the margin ranking.
    Advances in Neural Information Processing Systems, 33, 17044-17056.

    Returns:
        torch.Tensor: difference between correct class confidence and max other class confidence
    """
    mask_true_class = torch.logical_not(F.one_hot(labels, num_classes=eval_softmax.shape[-1]))
    labels = torch.unsqueeze(torch.broadcast_to(labels, eval_softmax.shape[:-1]), -1)
    correct = torch.take_along_dim(eval_softmax, labels, dim=-1).squeeze(-1)
    mar = correct - torch.max(eval_softmax * mask_true_class, dim=-1).values
    return mar


def error_l2_norm(eval_softmax: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """From
    Paul, M., Ganguli, S., and Dziugaite, G. K. (2021). Deep learning on a data
    diet: Finding important examples early in training. Advances in Neural In-
    formation Processing Systems, 34.

    Returns:
        torch.Tensor: EL2N score.
    """
    one_hot = F.one_hot(labels, num_classes=eval_softmax.shape[-1])
    error = eval_softmax - one_hot
    el2n = torch.sqrt(torch.sum(torch.square(error), dim=-1))
    return el2n
