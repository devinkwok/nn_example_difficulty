import torch
from scipy.stats import rankdata


def rank(metric: torch.Tensor):
    """Turns metrics into ranks (not differentiable). Smallest rank indicates smallest score.

    Args:
        metric (torch.Tensor): array of metrics with dimensions $(\dots, N)$

    Returns:
        torch.Tensor: array with same dimensions as metric,
            with values replaced by rank over last dimension $N$.
    """
    # use scipy.stats.rankdata to handle ties automatically
    # subtract 1 as rankdata starts from 1, not 0
    return torch.tensor(rankdata(metric, axis=-1) - 1)


def order_to_rank(argsort_idx: torch.Tensor):
    rank_idx = torch.arange(argsort_idx.shape[-1]).broadcast_to(argsort_idx.shape)
    ranks = torch.empty_like(argsort_idx)
    ranks = torch.scatter(ranks, dim=-1, index=argsort_idx, src=rank_idx)
    return ranks


#TODO complexity gap closed-form computation on data
