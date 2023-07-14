from abc import ABC
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np


__all__ = [
    "OnlineMean",
    "OnlineVariance",
]


class Accumulator(ABC):
    """Metadata must be a type that can be stored by numpy.savez without use_pickle=True.
        E.g. str, int, float
    Metadata must have keys distinct from variables in object,
        E.g. a key of `n` is not allowed
    However, metadata given to __init__() or add() can have the same keys and will remain distinct.
    """
    def __init__(self, dtype=torch.float64, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        # dtype allows using higher precision to minimize errors when accumulating
        self.metadata = {**metadata, "classname": type(self).__name__, "dtype": str(dtype), "device": str(device)}
        self.metadata_lists = defaultdict(list)
        for k, v in metadata_lists.items():
            assert isinstance(v, list)
            self.metadata_lists[k] = v

    @staticmethod
    def str_to_torch_dtype(dtype: str) -> torch.dtype:
        dtype = str(dtype)
        assert dtype.startswith("torch.")
        dtype = dtype.split("torch.")[1]
        return getattr(torch, dtype)

    @classmethod
    def load(cls, file: Path):
        load_dict = dict(np.load(file))
        metadata, data, lists = {}, {}, {}
        for k, v in load_dict.items():
            # strip prefixes
            prefix, key = k[:5], k[5:]
            if prefix == "meta_":  # unbox singleton from np.ndarray
                metadata[key] = v.item()
            elif prefix == "data_":
                data[key] = v
            elif prefix == "list_":
                lists[key] = v
        assert cls.__name__ == str(metadata["classname"])
        dtype = cls.str_to_torch_dtype(metadata["dtype"])
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.tensor(v, dtype=dtype, device=str(metadata["device"]))
        for k, v in lists.items():
            lists[k] = list(v)  # convert np.ndarrays back to lists
        return cls( **data, metadata_lists=lists, **metadata)

    @property
    def dtype(self):
        return self.str_to_torch_dtype(self.metadata["dtype"])

    @property
    def device(self):
        return str(self.metadata["device"])

    def save(self, file: Path, **data):
        # cannot have metadata that shares same keys as data, otherwise causes conflict when loading
        assert set(data.keys()).isdisjoint(set(self.metadata.keys()))
        # add prefixes so that metadata and data can be stored in same npz file
        # omit any None values, as None requires allow_pickle=True to load
        data_dict = {"data_" + k: v for k, v in data.items() if v is not None}
        metadata = {"meta_" + k: v for k, v in self.metadata.items() if v is not None}
        lists = {"list_" + k: v for k, v in self.metadata_lists.items() if v is not None}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.detach().cpu().numpy()
        np.savez(file, **data_dict, **metadata, **lists)

    def _add(self, *tensors: torch.Tensor, **metadata):
        for k, v in metadata.items():
            self.metadata_lists[k].append(v)
        output = tuple(x.to(dtype=self.dtype) for x in tensors)
        return output[0] if len(tensors) == 1 else output

    def add(self, x: torch.Tensor, dim=None, **metadata):
        """In place operation
        """
        raise NotImplementedError

    def get(self) -> torch.Tensor:
        """Returns a reference and NOT a copy,
        so will change with future calls to add()
        """
        raise NotImplementedError


class OnlineMean(Accumulator):
    def __init__(self, n=None, sum=None, dtype=torch.float64, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        if n is None:
            n = 0
        self.n = n
        self.sum = sum

    def save(self, file: Path):
        super().save(file, n=self.n, sum=self.sum)

    def add(self, x: torch.Tensor, dim=None, **metadata):
        x = super()._add(x, **metadata)
        if dim is None:
            self.n += 1
        else:
            self.n += x.shape[dim]
            x = torch.sum(x, dim=dim)
        if self.sum is None:
            self.sum = torch.zeros_like(x)
        self.sum += x
        return self

    def get(self):
        if self.sum is None:
            return None
        return self.sum / self.n


class OnlineVariance(Accumulator):
    """Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419-420.
    """
    def __init__(self, n=None, sum=None, sum_sq=None, dtype=torch.float64, device="cpu", metadata_lists: Dict[str, list]={}, **metadata):
        super().__init__(dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        self.mean = OnlineMean(n=n, sum=sum, dtype=dtype, device=device)
        self.sum_sq = sum_sq

    def save(self, file: Path):
        super().save(file, n=self.mean.n, sum=self.mean.sum, sum_sq=self.sum_sq)

    def add(self, x: torch.Tensor, dim=None, **metadata):
        x = super()._add(x, **metadata)
        prev_mean = self.mean.get()
        self.mean.add(x, dim=dim)
        curr_mean = self.mean.get()
        if prev_mean is None:  # only 1 sample
            prev_mean = torch.zeros_like(curr_mean)
            self.sum_sq = torch.zeros_like(curr_mean)
        if dim is not None:  # reshape to match x
            prev_mean = prev_mean.unsqueeze(dim).broadcast_to(x.shape)
            curr_mean = curr_mean.unsqueeze(dim).broadcast_to(x.shape)
            self.sum_sq += torch.sum((x - curr_mean) * (x - prev_mean), dim=dim)
        else:
            self.sum_sq += (x - curr_mean) * (x - prev_mean)
        return self

    def get(self):
        if self.sum_sq is None:
            return None
        return self.sum_sq / (self.mean.n - 1)


class BatchAccumulator(Accumulator):
    """
    Assumes batches are indexed over dim=-1
    Note: select_subset() clones tensors, so may not be differentiable
    """
    def __init__(self,
        n_items: int,  # must specify n_items at first init
        n=None,
        dtype=torch.float64,
        device="cpu",
        metadata_lists: Dict[str, list]={},
        **metadata,
    ):
        super().__init__(n_items=n_items, dtype=dtype, device=device, metadata_lists=metadata_lists, **metadata)
        self.n = torch.zeros(n_items, dtype=torch.long, device=device) if n is None else n

    def save(self, file: Path, **data):
        super().save(file, n=self.n, **data)

    def add(self, *tensors: List[torch.Tensor], minibatch_idx: torch.Tensor=None, **metadata):
        fixed_dtype = super()._add(*tensors, **metadata)
        if len(tensors) == 1:  # super()._add() automatically unpacks singleton tuples, need to re-pack for select_subset()
            fixed_dtype = (fixed_dtype,)
        self.n[minibatch_idx] += 1
        return self.select_subset(*fixed_dtype, minibatch_idx=minibatch_idx)

    def select_subset(self, *tensors, minibatch_idx=None):
        if minibatch_idx is None:
            minibatch_idx = torch.arange(len(self.n), device=self.device)
        output = tuple(x.index_select(dim=-1, index=minibatch_idx) for x in tensors)
        return output[0] if len(tensors) == 1 else output

    def update_subset_(self, target: torch.Tensor, source: torch.Tensor, minibatch_idx: torch.Tensor=None):
        if minibatch_idx is None:
            minibatch_idx = torch.arange(len(self.n), device=self.device)
        target[..., minibatch_idx] = source  # in place operation
