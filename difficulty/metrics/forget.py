"""Metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.

Default dimensions are (..., T, N) where T is steps (iterations)
"""
import torch
from typing import Dict

from difficulty.utils import detach_tensors


__all__ = [
    "forget_metrics",
    "first_learn",
    "first_unforgettable",
    "first_forget",
    "first_unlearnable",
    "count_forgetting",
    "is_unforgettable",
]


def forget_metrics(accuracy: torch.Tensor, start_at_zero=True, dim=-2, detach=True, to_cpu=True, to_numpy=False) -> Dict[str, torch.Tensor]:
    count_metrics = {"countforget": count_forgetting(accuracy, start_at_zero=start_at_zero, dim=dim),
                     "unforgettable": is_unforgettable(accuracy, start_at_zero=start_at_zero, dim=dim)}
    if start_at_zero:
        order_metrics = {"firstlearn": first_learn(accuracy, dim=dim),
                         "firstunforgettable": first_unforgettable(accuracy, dim=dim)}
    else:
        order_metrics = {"firstforget": first_forget(accuracy, dim=dim),
                         "firstunlearnable": first_unlearnable(accuracy, dim=dim)}
    return detach_tensors({**count_metrics, **order_metrics}, to_cpu=to_cpu, to_numpy=to_numpy)


def _concat_iter(tensor: torch.Tensor, fill_value=0, dim: int=-2, at_end=False):
    single = torch.index_select(tensor, dim=dim, index=torch.tensor([0], dtype=torch.long, device=tensor.device))
    single = torch.full_like(single, fill_value)
    combined = [tensor, single] if at_end else [single, tensor]
    return torch.cat(combined, dim=dim)


def _change_events(zero_one_accuracy: torch.Tensor, falling_edge: bool, dim: int=-2) -> torch.Tensor:
    prev = torch.roll(zero_one_accuracy, 1, dims=dim)  # (0,1,2, ..., -1) -> (-1,0,1, ..., -2)
    if falling_edge:  # events transitioning from 1 to 0
        events = torch.logical_and(prev, torch.logical_not(zero_one_accuracy))
    else:  # 0 to 1 (always guaranteed to occur at least once)
        events = torch.logical_and(torch.logical_not(prev), zero_one_accuracy)
    # exclude first event, which is the transition from -1 to 0 in zero_one_accuracy
    idx = torch.arange(1, zero_one_accuracy.shape[dim], device=zero_one_accuracy.device)
    return torch.index_select(events, dim=dim, index=idx)


def _forgetting_events(zero_one_accuracy: torch.Tensor, start_at_zero: bool=True, dim: int=-2) -> torch.Tensor:
    """Forgetting events, defined as falling edges for zero-one-accuracy (i.e. a transition from 1 to 0).

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T, N) where T is the step
        start_at_zero (bool, optional): whether to assume events begin at 0 (untrained) or 1 (trained).
            In particular, if False it is possible to have a forgetting event at the first timestep
            (e.g. if all T have zero accuracy, there is a forgetting event at T=0). Defaults to True.

    Returns:
        torch.Tensor: boolean tensor of shape (..., T, N) where True indicates a forgetting event.
    """
    acc = _concat_iter(zero_one_accuracy, False if start_at_zero else True, dim=dim)
    return _change_events(acc, falling_edge=True, dim=dim)


def _learning_events(zero_one_accuracy: torch.Tensor, start_at_zero: bool=True, dim: int=-2) -> torch.Tensor:
    """Learning events, defined as rising edges for zero-one-accuracy (i.e. a transition from 0 to 1).

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T, N) where T is the step
        start_at_zero (bool, optional): whether to assume events begin at 0 (untrained) or 1 (trained).
            In particular, if True it is possible to have a learning event at the first timestep
            (e.g. if all T have one accuracy, there is a learning event at T=0). Defaults to True.

    Returns:
        torch.Tensor: boolean tensor of shape (..., T, N) where True indicates a forgetting event.
    """
    acc = _concat_iter(zero_one_accuracy, False if start_at_zero else True, dim=dim)
    return _change_events(acc, falling_edge=False, dim=dim)


def _first_event_time(events: torch.Tensor, dim=-2) -> torch.Tensor:
    # force match at 0 or T if there are no matches
    # multiply by 1 to make numeric to allow use of argmax
    events_plus_end = 1*_concat_iter(events, True, dim=dim, at_end=True)
    # from torch.argmax: In case of multiple occurrences of the maximum values
    # the indices corresponding to the first occurrence are returned.
    first_event = torch.argmax(events_plus_end, dim=dim)
    return first_event


def _last_event_time(events: torch.Tensor, dim=-2) -> torch.Tensor:
    reversed = torch.flip(events, dims=[dim])
    backward_idx = _first_event_time(reversed, dim)
    # _first_event_time returns events.shape[dim] if no events
    # reversing the idx makes this become -1, so take modulo to map back to events.shape[dim]
    return (events.shape[dim] - 1 - backward_idx) % (events.shape[dim] + 1)


def first_learn(zero_one_accuracy: torch.Tensor, dim=-2) -> torch.Tensor:
    """Equivalent to _first_event_time(learning_events(zero_one_accuracy, start_at_zero=True, dim=dim), dim=dim)

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T, N) where T is the step

    Returns:
        torch.Tensor: iteration at which an example is first learned,
            from 0 to T inclusive where T means never learned
    """
    return _first_event_time(_learning_events(
        zero_one_accuracy, start_at_zero=True, dim=dim), dim=dim)


def first_unforgettable(zero_one_accuracy: torch.Tensor, dim=-2) -> torch.Tensor:
    """Equivalent to _last_event_time(forgetting_events(_concat_iter(zero_one_accuracy, True, dim=dim, at_end=True), start_at_zero=True, dim=dim), dim=dim)
    Also called iteration learned in Baldock et al. (2021),
    and consistently-learned in Siddiqui et al. (2022).

    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Siddiqui, S. A., Rajkumar, N., Maharaj, T., Krueger, D., & Hooker, S. (2022).
    Metadata archaeology: Unearthing data subsets by leveraging training dynamics.
    arXiv preprint arXiv:2209.10015.

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step

    Returns:
        torch.Tensor: step after which classification is always correct,
            from 0 to T inclusive where 0 means always learned and T means never learned
    """
    guarantee_learn_at_end = _concat_iter(zero_one_accuracy, True, dim=dim, at_end=True)
    return _last_event_time(_learning_events(guarantee_learn_at_end, start_at_zero=True, dim=dim), dim=dim)


def first_forget(zero_one_accuracy: torch.Tensor, dim=-2) -> torch.Tensor:
    """Equivalent to _first_event_time(forgetting_events(zero_one_accuracy, start_at_zero=False, dim=dim), dim=dim)

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T, N) where T is the step

    Returns:
        torch.Tensor: iteration at which an example is first forgotten,
            from 0 to T inclusive where T means never learned
    """
    return _first_event_time(_forgetting_events(zero_one_accuracy, start_at_zero=False, dim=dim), dim=dim)


def first_unlearnable(zero_one_accuracy: torch.Tensor, dim=-2) -> torch.Tensor:
    """Equivalent to _last_event_time(forgetting_events(_concat_iter(zero_one_accuracy, False, dim=dim, at_end=True), start_at_zero=False, dim=dim), dim=dim)

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T, N) where T is the step

    Returns:
        torch.Tensor: step after which classification is always wrong
    """
    guarantee_forget_at_end = _concat_iter(zero_one_accuracy, False, dim=dim, at_end=True)
    return _last_event_time(_forgetting_events(guarantee_forget_at_end, start_at_zero=False, dim=dim), dim=dim)


def count_forgetting(zero_one_accuracy: torch.Tensor, start_at_zero=True, dim: int=-2) -> torch.Tensor:
    """Equivalent to torch.count_nonzero(forgetting_events(zero_one_accuracy, start_at_zero=start_at_zero, dim=dim), dim=dim)

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step
        start_at_zero (bool, optional): whether to assume events begin at 0 (untrained) or 1 (trained).
            In particular, if False it is possible to have a forgetting event at the first timestep
            (e.g. if all T have zero accuracy, there is a forgetting event at T=0). Defaults to True.

    Returns:
        torch.Tensor: number of forgetting events (..., N)
    """
    return torch.count_nonzero(_forgetting_events(zero_one_accuracy, start_at_zero=start_at_zero, dim=dim), dim=dim)


def is_unforgettable(zero_one_accuracy: torch.Tensor, start_at_zero=True, dim: int=-2) -> torch.Tensor:
    """
    An example is unforgettable if it is 1) learned, and 2) never forgotten
    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step
        start_at_zero (bool, optional): whether to assume events begin at 0 (untrained) or 1 (trained).
            In particular, if False it is possible to have a forgetting event at the first timestep
            (e.g. if all T have zero accuracy, there is a forgetting event at T=0). Defaults to True.

    Returns:
        torch.Tensor: 0 if example is forgotten, 1 if example is not forgotten (..., N)
    """
    unforgotten = count_forgetting(zero_one_accuracy, start_at_zero=start_at_zero, dim=dim) == 0
    is_learned = first_learn(zero_one_accuracy, dim=dim) < zero_one_accuracy.shape[dim]
    return torch.logical_and(unforgotten, is_learned)
