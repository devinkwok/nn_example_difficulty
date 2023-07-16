"""Online metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.

Following functions assume dimensions (N,) where N is examples
"""
from pathlib import Path
from typing import Dict, List, Tuple
import torch

from difficulty.metrics.forget import count_forgetting
from difficulty.metrics.accumulator import BatchAccumulator


__all__ = [
    "OnlineFirstLearn",
    "OnlineFirstUnforgettable",
    # "OnlineFirstForget",
    # "OnlineFirstUnlearnable",
    "OnlineCountForgetting",
    "OnlineIsUnforgettable",
]


class OnlineForgetting(BatchAccumulator):

    def init_tensors(self, zero_one_accuracy, tensors_defaults: List[Tuple[torch.Tensor, float]]):
        zero_one_accuracy = zero_one_accuracy.to(dtype=self.dtype, device=self.device)
        # fix shape to have n_items
        shape = list(zero_one_accuracy.shape)
        shape[-1] = self.get_metadata("n_items")
        outputs = []
        for tensor, default in tensors_defaults:
            if tensor is None:
                outputs.append(torch.full(shape, default, dtype=self.dtype, device=self.device))
            else:
                outputs.append(tensor)
        return zero_one_accuracy, *outputs


class OnlineFirstLearn(OnlineForgetting):
    def __init__(self, n_items: int=None, n=None, learn_time=None, is_learned=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        self.learn_time = learn_time
        self.is_learned = is_learned

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy, self.learn_time, self.is_learned = self.init_tensors(
            zero_one_accuracy, [(self.learn_time, 0), (self.is_learned, 0)])
        learn_time, is_learned = super().add(self.learn_time, self.is_learned, minibatch_idx=minibatch_idx, **metadata)
        is_learned = torch.logical_or(is_learned, zero_one_accuracy).to(self.dtype)
        learn_time += torch.logical_not(is_learned) * 1
        self.update_subset_(self.learn_time, learn_time, minibatch_idx=minibatch_idx)
        self.update_subset_(self.is_learned, is_learned, minibatch_idx=minibatch_idx)

    def save(self, file: Path):
        super().save(file, is_learned=self.is_learned, learn_time=self.learn_time)

    def get(self) -> torch.Tensor:
        return self.learn_time


class OnlineFirstUnforgettable(OnlineForgetting):
    def __init__(self, n_items: int=None, n=None, consecutive_learned=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        self.consecutive_learned = consecutive_learned

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy, self.consecutive_learned = self.init_tensors(
            zero_one_accuracy, [(self.consecutive_learned, 0)])
        consecutive_learned = super().add(self.consecutive_learned, minibatch_idx=minibatch_idx, **metadata)
        consecutive_learned += 1
        consecutive_learned *= zero_one_accuracy  # reset to 0 if not learned
        self.update_subset_(self.consecutive_learned, consecutive_learned, minibatch_idx=minibatch_idx)

    def save(self, file: Path):
        super().save(file, consecutive_learned=self.consecutive_learned)

    def get(self) -> torch.Tensor:
        return self.n - self.consecutive_learned


#TODO OnlineFirstForget


#TODO OnlineFirstUnlearnable


class OnlineCountForgetting(OnlineForgetting):

    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, n_forget=None, prev_acc=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.n_forget = n_forget

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, n_forget=self.n_forget)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy, self.prev_acc, self.n_forget = self.init_tensors(
            zero_one_accuracy, [(self.prev_acc, 0 if self.get_metadata("start_at_zero") else 1), (self.n_forget, 0)])
        n_forget, prev_acc = super().add(self.n_forget, self.prev_acc, minibatch_idx=minibatch_idx, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=-2)
        n_forget += count_forgetting(acc, start_at_zero=True, dim=-2)
        self.update_subset_(self.n_forget, n_forget, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return self.n_forget


class OnlineIsUnforgettable(OnlineForgetting):

    def __init__(self, n_items: int=None, start_at_zero: bool=True, n=None, unforgotten=None, prev_acc=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(n_items, n=n, dtype=dtype, device=device, metadata_lists=metadata_lists, start_at_zero=start_at_zero, **metadata)
        self.prev_acc = prev_acc
        self.unforgotten = unforgotten

    def save(self, file: Path):
        super().save(file, prev_acc=self.prev_acc, unforgotten=self.unforgotten)

    def add(self, zero_one_accuracy: torch.Tensor, minibatch_idx: torch.Tensor=None, **metadata):
        zero_one_accuracy, self.prev_acc, self.unforgotten = self.init_tensors(
            zero_one_accuracy, [(self.prev_acc, 0 if self.get_metadata("start_at_zero") else 1), (self.unforgotten, 1)])
        unforgotten, prev_acc = super().add(self.unforgotten, self.prev_acc, minibatch_idx=minibatch_idx, **metadata)
        acc = torch.stack([prev_acc, zero_one_accuracy], dim=0)
        no_forget = count_forgetting(acc, start_at_zero=True, dim=0) == 0
        unforgotten = torch.logical_and(unforgotten, no_forget).to(dtype=self.dtype)
        self.update_subset_(self.unforgotten, unforgotten, minibatch_idx=minibatch_idx)
        self.update_subset_(self.prev_acc, zero_one_accuracy, minibatch_idx=minibatch_idx)

    def get(self) -> torch.Tensor:
        return torch.logical_and(self.unforgotten, self.prev_acc)
