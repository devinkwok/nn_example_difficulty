"""Metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.

Following functions assume dimensions (..., T, N) where T is steps (iterations)
"""
from pathlib import Path
from typing import Dict
import torch

from difficulty.metrics.accumulator import BatchAccumulator


__all__ = [
    "first_learn",
    "first_unforgettable",
    "first_forget",
    "first_unlearnable",
    "count_forgetting",
    "is_unforgettable",
    # "OnlineFirstLearn",
    # "OnlineFirstUnforgettable",
    # "OnlineFirstForget",
    # "OnlineFirstUnlearnable",
    "OnlineCountForgetting",
    # "OnlineIsUnforgettable",
]


def _concat_iter(tensor: torch.Tensor, fill_value=0, dim: int=-2, at_end=False):
    single = torch.index_select(tensor, dim=dim, index=torch.tensor([0]))
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
    idx = torch.arange(1, zero_one_accuracy.shape[dim])
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

    Also called iteration learned in
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

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


class OnlineFirstLearn(BatchAccumulator):
    #TODO test
    def __init__(self, n_items: int=None, n=None, learn_time=None, is_learned=None, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items=n_items, n=n, device=device, metadata_lists=metadata_lists, **metadata)
        self.is_learned = torch.zeros(n_items, dtype=torch.bool) if is_learned is None else is_learned
        self.learn_time = torch.zeros(n_items, dtype=torch.long) if learn_time is None else learn_time

    def save(self, file: Path):
        super().save(file, is_learned=self.is_learned, learn_time=self.learn_time)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        learn_time, is_learned = super().add(minibatch_idx, self.learn_time, self.is_learned, **metadata)
        is_learned = torch.logical_or(is_learned, zero_one_accuracy)
        learn_time += torch.logical_not(is_learned) * 1
        self.update_subset_(self.learn_time, learn_time, minibatch_idx=minibatch_idx)
        self.update_subset_(self.is_learned, is_learned, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return self.learn_time


#TODO OnlineFirstUnforgettable


#TODO OnlineFirstForget


#TODO OnlineFirstUnlearnable


class OnlineCountForgetting(BatchAccumulator):

    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, n_forget=None, prev_acc=None, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.n_forget = n_forget

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, n_forget=self.n_forget)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        if self.prev_acc is None:
            # fix shape to have n_items
            shape = list(zero_one_accuracy.shape)
            shape[-1] = self.metadata["n_items"]
            self.prev_acc = torch.full(shape, 0 if self.metadata["start_at_zero"] else 1, dtype=torch.bool)
            self.n_forget = torch.zeros(shape, dtype=torch.long)
        n_forget, prev_acc = super().add(self.n_forget, self.prev_acc, minibatch_idx=minibatch_idx, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=-2)
        n_forget += count_forgetting(acc, start_at_zero=True, dim=-2)
        self.update_subset_(self.n_forget, n_forget, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return self.n_forget


class OnlineIsUnforgettable(BatchAccumulator):
    #TODO test
    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, unforgotten=None, prev_acc=None, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.unforgotten = unforgotten

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, unforgotten=self.unforgotten)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        if self.prev_acc is None:
            # fix shape to have n_items
            shape = list(zero_one_accuracy.shape)
            shape[-1] = self.metadata["n_items"]
            self.prev_acc = torch.full(shape, 0 if self.metadata["start_at_zero"] else 1, dtype=torch.bool)
            self.unforgotten = torch.zeros(shape, dtype=torch.bool)
        unforgotten, prev_acc = super().add(minibatch_idx, self.unforgotten, self.prev_acc, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=0)
        no_forget = count_forgetting(acc, start_at_zero=True, dim=0) == 0
        unforgotten = torch.logical_and(unforgotten, no_forget)
        self.update_subset_(self.unforgotten, unforgotten, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return torch.logical_and(self.unforgotten, self.prev_acc)
