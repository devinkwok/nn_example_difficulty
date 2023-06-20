import torch


def rank(metric: torch.Tensor):
    """Turns metrics into ranks.

    Args:
        metric (torch.Tensor): array of metrics with dimensions $(\dots, N)$

    Returns:
        torch.Tensor: array with same dimensions as metric,
            with values replaced by rank over last dimension $N$.
    """
    sorted_idx = torch.argsort(metric, dim=-1)
    rank_idx = torch.arange(metric.shape[-1]).broadcast_to(sorted_idx.shape)
    ranks = torch.empty_like(sorted_idx)
    ranks = torch.scatter(ranks, dim=-1, index=sorted_idx, src=rank_idx)
    return ranks


#TODO complexity gap closed-form computation on data
