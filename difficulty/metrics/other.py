import numpy as np


def rank(metric: np.ndarray):
    """Turns metrics into ranks.

    Args:
        metric (np.ndarray): array of metrics with dimensions $(\dots, N)$

    Returns:
        np.ndarray: array with same dimensions as metric,
            with values replaced by rank over last dimension $N$.
    """
    sorted_idx = np.argsort(metric, axis=-1)
    rank_idx = np.arange(metric.shape[-1])
    ranks = np.empty_like(sorted_idx)
    np.put_along_axis(ranks, sorted_idx, rank_idx, axis=-1)
    return ranks
