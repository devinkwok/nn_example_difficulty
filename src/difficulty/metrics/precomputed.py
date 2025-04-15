# metrics that are computed over multiple runs
from pathlib import Path
import numpy as np
import torch

from difficulty.metrics.other import order_to_rank


__all__ = [
    "load_adv",
    "load_ret",
    "load_agr",
    "load_conf",
    "load_priv",
    "load_influence_memorization",
    "load_memorization",
    "load_self_supervised_prototypes",
]


def _precomputed_path(subdirectory):
    return Path(__file__).parent.parent / "precomputed" / subdirectory


def _load_carlini(key: str, dataset: str, train=True):
    path = _precomputed_path("carlini-et-al-metrics-order")
    assert key in set(["adv", "agr", "conf", "priv", "ret"])
    datasets = {"cifar10": "cifar", "fashionmnist": "fashion", "mnist": "mnist", "imagenet": "imagenet"}
    if dataset not in datasets:
        raise ValueError(f"Unrecognized dataset {dataset}")
    data = np.load(path / f"order_{datasets[dataset]}_{key}_{'train' if train else 'test'}.npy")
    # change from order to rank
    return order_to_rank(torch.tensor(data, dtype=torch.long))


def load_adv(dataset: str, train=True):
    # get path to data relative to this file
    return _load_carlini("adv", dataset, train=train)


def load_ret(dataset: str, train=True):
    """
    Load precomputed holdout retraining scores, which is the symmetric KL-divergence between a model
    with the example in the training set, and the same model finetuned further on a held-out dataset.

    Carlini, N., Erlingsson, U., & Papernot, N. (2019).
    Distribution density, tails, and outliers in machine learning: Metrics and applications.
    arXiv preprint arXiv:1910.13427.
    """
    return _load_carlini("ret", dataset, train=train)


def load_agr(dataset: str, train=True):
    """
    Load precomputed ensemble agreeableness scores,
    which are the Jensen-Shannon divergence between the outputs of all model pairs ini an ensemble:

    Carlini, N., Erlingsson, U., & Papernot, N. (2019).
    Distribution density, tails, and outliers in machine learning: Metrics and applications.
    arXiv preprint arXiv:1910.13427.
    """
    return _load_carlini("agr", dataset, train=train)


def load_conf(dataset: str, train=True):
    """
    Load precomputed ensemble max confidence scores from:

    Carlini, N., Erlingsson, U., & Papernot, N. (2019).
    Distribution density, tails, and outliers in machine learning: Metrics and applications.
    arXiv preprint arXiv:1910.13427.

    This is identical to applying difficulty.metrics.max_confidence() over multiple models at the end of training.
    """
    return _load_carlini("conf", dataset, train=train)


def load_priv(dataset: str, train=True):
    """
    Load precomputed differential privacy scores, which is the minimum epsilon at which
    epsilon-differentially-private SGD correctly classifies the example at least 90% of the time.

    Carlini, N., Erlingsson, U., & Papernot, N. (2019).
    Distribution density, tails, and outliers in machine learning: Metrics and applications.
    arXiv preprint arXiv:1910.13427.
    """
    return _load_carlini("priv", dataset, train=train)


def load_influence_memorization(dataset="cifar100"):
    """
    Load precomputed influence and memorization scores from:

    Feldman, V., & Zhang, C. (2020).
    What neural networks memorize and why: Discovering the long tail via influence estimation.
    Advances in Neural Information Processing Systems, 33, 2881-2891.

    Source: https://pluskid.github.io/influence-memorization/data/cifar100_high_infl_pairs_infl0.15_mem0.25.npz
    """
    path = _precomputed_path("feldman-zhang-influence-memorization")
    if dataset == "cifar100":
        data = np.load(path / "cifar100_high_infl_pairs_infl0.15_mem0.25.npz")
        return data['tr_idx'], data['tt_idx'], data['infl'], data['mem']
    else:
        raise ValueError(f"Unrecognized dataset {dataset}")


def _load_precomputed(directory, file_template, **kwargs):
    path = _precomputed_path(directory)
    format_values = {}
    for k, (arg, allowed_args) in kwargs.items():
        if arg not in allowed_args:
            raise ValueError(f"Unrecognized {k}: {arg}")
        format_values[k] = arg
    return np.load(path / file_template.format(**format_values))['arr_0']


def load_memorization(dataset):
    """
    Load precomputed influence and memorization scores from:

    Feldman, V., & Zhang, C. (2020).
    What neural networks memorize and why: Discovering the long tail via influence estimation.
    Advances in Neural Information Processing Systems, 33, 2881-2891.

    Source: https://github.com/google-research/heldout-influence-estimation
    """
    return _load_precomputed(
        "feldman-zhang-influence-memorization",
        "memorization-{dataset}-0.7.npz",
        dataset=(dataset, {"cifar100"}),
    )


def load_self_supervised_prototypes(dataset, model="swav"):
    """
    Load precomputed influence and memorization scores from:

    Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. (2022).
    Beyond neural scaling laws: beating power law scaling via data pruning.
    Advances in Neural Information Processing Systems, 35, 19523-19536.

    Model source for "swav": https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
    """
    return _load_precomputed(
        "sorscher-selfproto", "selfproto-{dataset}-{model}.npz",
        dataset=(dataset, {"cifar10", "cifar100"}),
        model=(model, {"swav"}),
    )
