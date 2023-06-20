from pathlib import Path
from collections import defaultdict
from typing import Dict

import torch
import numpy as np


class Accumulator:
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
        save_dict = {**data, **self.metadata}
        for k, v in save_dict.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.detach().cpu().numpy()
        np.savez(file, **save_dict)

    def add(self, x: torch.Tensor, dim=None, **metadata):
        data = self.metadata.copy()
        for k, v in metadata.items():
            data[k].append(v)
        x = x.to(dtype=self.dtype)
        return x, data

    def get(self):
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
        x, metadata = super().add(x, dim, **metadata)
        if dim is None:
            n = self.n + 1
        else:
            n = self.n + x.shape[dim]
            x = torch.mean(x, dim=dim)
        if self.mean is None:
            mean = x
        else:
            ratio = self.n / n
            mean = self.mean + (x - self.mean) * ratio
        return OnlineMean(n=n, mean=mean, **metadata)

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
        x, metadata = super().add(x, dim, **metadata)
        prev_mean = self.mean.get()
        mean_obj = self.mean.add(x, dim=dim)
        curr_mean = mean_obj.get()
        n = mean_obj.n
        if prev_mean is None:  # only 1 sample
            prev_mean = torch.zeros_like(curr_mean)
        if dim is not None:  # reshape to match x
            prev_mean = prev_mean.unsqueeze(dim).broadcast_to(x.shape)
            curr_mean = curr_mean.unsqueeze(dim).broadcast_to(x.shape)
            sum_sq = (x - curr_mean) * (x - prev_mean)
            sum_sq = torch.sum(sum_sq, dim=dim)
        else:
            sum_sq = (x - curr_mean) * (x - prev_mean)
        if self.sum_sq is not None:  # more than 1 sample
            sum_sq += self.sum_sq
        return OnlineVariance(sum_sq=sum_sq, mean=mean_obj.mean, n=mean_obj.n, **metadata)

    def get(self):
        return self.sum_sq / (self.mean.n - 1)
