"""Following functions assume the last dimension of eval_logits or eval_softmax
is over $C$ classes, and second last dimension is over $N$ examples.
"""
import numpy as np
import torch
import torch.nn.functional as F


def softmax(eval_logits: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        softmax = F.softmax(torch.tensor(eval_logits), dim=-1)
    return softmax.numpy()


def zero_one_accuracy(eval_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.argmax(eval_logits, axis=-1) == labels


def entropy(eval_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # broadcast labels
    labels = np.broadcast_to(labels, eval_logits.shape[:-1])
    labels = np.moveaxis(labels, -1, 0)
    eval_logits = np.moveaxis(eval_logits, [-2, -1], [0, 1])
    # make into tensors to use pytorch functional cross_entropy
    labels = torch.tensor(labels)
    eval_logits = torch.tensor(eval_logits)
    with torch.no_grad():
        entropy = F.cross_entropy(eval_logits, labels, reduction='none')
    # move dims back
    entropy = np.moveaxis(entropy.numpy(), 0, -1)
    return entropy


def class_confidence(eval_softmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.expand_dims(np.broadcast_to(labels, eval_softmax.shape[:-1]), -1)
    return np.take_along_axis(eval_softmax, labels, axis=-1).squeeze(-1)


def max_confidence(eval_softmax: np.ndarray) -> np.ndarray:
    return np.max(eval_softmax, axis=-1)


def margin(eval_softmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """From
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.

    Returns:
        np.ndarray: difference between correct class confidence and max other class confidence
    """
    labels = np.expand_dims(np.broadcast_to(labels, eval_softmax.shape[:-1]), -1)
    correct = np.take_along_axis(eval_softmax, labels, axis=-1).squeeze(-1)
    max_other = np.copy(eval_softmax)
    np.put_along_axis(max_other, labels, -1., axis=-1)
    return correct - np.max(max_other, axis=-1)


def error_l2_norm(eval_softmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """From
    Paul, M., Ganguli, S., and Dziugaite, G. K. (2021). Deep learning on a data
    diet: Finding important examples early in training. Advances in Neural In-
    formation Processing Systems, 34.

    Returns:
        np.ndarray: EL2N score.
    """
    one_hot = F.one_hot(torch.tensor(labels), num_classes=eval_softmax.shape[-1])
    error = eval_softmax - one_hot.numpy()
    return np.sqrt(np.sum(np.square(error), axis=-1))
