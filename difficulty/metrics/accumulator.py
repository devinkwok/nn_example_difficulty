from abc import ABC
from pathlib import Path
from collections import defaultdict
from typing import Dict

import torch
import numpy as np


__all__ = [
    "OnlineMean",
    "OnlineVariance",
]


class Accumulator(ABC):
    def __init__(self, dtype=torch.float64, **metadata_lists: Dict[str, list]):
        self.dtype = dtype  # use higher precision to minimize errors when accumulating
        self.metadata = defaultdict(list)
        for k, v in metadata_lists.items():
            assert isinstance(v, list)
            self.metadata[k] = v

    @classmethod
    def load(cls, file: Path):
        load_dict = dict(np.load(file))
        for k, v in load_dict.items():
            if isinstance(v, np.ndarray):
                load_dict[k] = torch.tensor(v)
        return cls(**load_dict)

    def save(self, file: Path, **data):
        # cannot have metadata that shares same keys as data
        assert set(data.keys()).isdisjoint(set(self.metadata.keys()))
        save_dict = {**data, **self.metadata}
        for k, v in save_dict.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.detach().cpu().numpy()
        np.savez(file, **save_dict)

    def _add(self, x: torch.Tensor, **metadata):
        for k, v in metadata.items():
            self.metadata[k].append(v)
        x = x.to(dtype=self.dtype)
        return x

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
    def __init__(self, n=None, mean=None, dtype=torch.float64, **metadata_lists: Dict[str, list]):
        super().__init__(dtype=dtype, **metadata_lists)
        if n is None:
            n = 0
        self.n = n
        self.mean = mean

    def save(self, file: Path):
        super().save(file, n=self.n, mean=self.mean)

    def add(self, x: torch.Tensor, dim=None, **metadata):
        """
            Let p be previous mean, q be mean of new entries,
            m be number of previous entries, and n be number of new entries.
            Then the new mean is:
            (p*m + q*n) / (m + n)
            = (p*(m + n) + (q-p)*n) / (m + n)
            = p + n / (m + n) * (q - p)
        """
        x = super()._add(x, **metadata)
        prev_n = self.n
        if dim is None:
            self.n += 1
        else:
            self.n += x.shape[dim]
            x = torch.mean(x, dim=dim)
        if self.mean is None:
            self.mean = x
        else:
            ratio = prev_n / self.n
            self.mean += (x - self.mean) * ratio
        return self

    def get(self):
        return self.mean


class OnlineVariance(Accumulator):
    """Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics, 4(3), 419-420.
    """
    def __init__(self, n=None, mean=None, sum_sq=None, dtype=torch.float64, **metadata_lists: Dict[str, list]):
        super().__init__(dtype=dtype, **metadata_lists)
        self.mean = OnlineMean(n=n, mean=mean)
        self.sum_sq = sum_sq

    def save(self, file: Path):
        super().save(file, n=self.mean.n, mean=self.mean.mean, sum_sq=self.sum_sq)

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
        return self.sum_sq / (self.mean.n - 1)
