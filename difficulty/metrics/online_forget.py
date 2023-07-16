"""Online metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.
"""
from pathlib import Path
from typing import Dict
import torch

from difficulty.metrics.forget import count_forgetting
from difficulty.metrics.accumulator import BatchAccumulator


__all__ = [
    "OnlineFirstLearn",
    # "OnlineFirstUnforgettable",
    # "OnlineFirstForget",
    # "OnlineFirstUnlearnable",
    "OnlineCountForgetting",
    "OnlineIsUnforgettable",
]


class OnlineFirstLearn(BatchAccumulator):
    def __init__(self, n_items: int=None, n=None, learn_time=None, is_learned=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        self.learn_time = learn_time
        self.is_learned = is_learned

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy = zero_one_accuracy.to(dtype=self.dtype, device=self.device)
        if self.learn_time is None:
            # fix shape to have n_items
            shape = list(zero_one_accuracy.shape)
            shape[-1] = self.metadata["n_items"]
            self.learn_time = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.is_learned = torch.zeros(shape, dtype=self.dtype, device=self.device)
        learn_time, is_learned = super().add(self.learn_time, self.is_learned, minibatch_idx=minibatch_idx, **metadata)
        is_learned = torch.logical_or(is_learned, zero_one_accuracy).to(self.dtype)
        learn_time += torch.logical_not(is_learned) * 1
        self.update_subset_(self.learn_time, learn_time, minibatch_idx=minibatch_idx)
        self.update_subset_(self.is_learned, is_learned, minibatch_idx=minibatch_idx)

    def save(self, file: Path):
        super().save(file, is_learned=self.is_learned, learn_time=self.learn_time)

    def get(self) -> torch.Tensor:
        return self.learn_time


#TODO OnlineFirstUnforgettable


#TODO OnlineFirstForget


#TODO OnlineFirstUnlearnable


class OnlineCountForgetting(BatchAccumulator):

    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, n_forget=None, prev_acc=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.n_forget = n_forget

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, n_forget=self.n_forget)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy = zero_one_accuracy.to(dtype=self.dtype, device=self.device)
        if self.prev_acc is None:
            # fix shape to have n_items
            shape = list(zero_one_accuracy.shape)
            shape[-1] = self.metadata["n_items"]
            self.prev_acc = torch.full(shape, 0 if self.metadata["start_at_zero"] else 1, dtype=self.dtype, device=self.device)
            self.n_forget = torch.zeros(shape, dtype=self.dtype, device=self.device)
        n_forget, prev_acc = super().add(self.n_forget, self.prev_acc, minibatch_idx=minibatch_idx, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=-2)
        n_forget += count_forgetting(acc, start_at_zero=True, dim=-2)
        self.update_subset_(self.n_forget, n_forget, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return self.n_forget


class OnlineIsUnforgettable(BatchAccumulator):

    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, unforgotten=None, prev_acc=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.unforgotten = unforgotten

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, unforgotten=self.unforgotten)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy = zero_one_accuracy.to(dtype=self.dtype, device=self.device)
        if self.prev_acc is None:
            # fix shape to have n_items
            shape = list(zero_one_accuracy.shape)
            shape[-1] = self.metadata["n_items"]
            self.prev_acc = torch.full(shape, 0 if self.metadata["start_at_zero"] else 1, dtype=self.dtype, device=self.device)
            self.unforgotten = torch.ones(shape, dtype=self.dtype, device=self.device)
        unforgotten, prev_acc = super().add(self.unforgotten, self.prev_acc, minibatch_idx=minibatch_idx, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=0)
        no_forget = count_forgetting(acc, start_at_zero=True, dim=0) == 0
        unforgotten = torch.logical_and(unforgotten, no_forget).to(dtype=self.dtype)
        self.update_subset_(self.unforgotten, unforgotten, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return torch.logical_and(self.unforgotten, self.prev_acc)
