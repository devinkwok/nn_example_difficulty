import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict, List
import numpy as np
import pandas as pd
import torch


class Stopwatch:
    def __init__(self, name="STOPWATCH"):
        self.name = name
        self.n_total = 0
        self.total_time = 0
        self.n_since_last_print = 0
        self.time_since_last_print = 0
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.n_total += 1
        self.n_since_last_print += 1
        self.start_time = time.time()

    def stop(self, message=None):
        self.stop_time = time.time()
        elapsed = self.stop_time - self.start_time
        self.total_time += elapsed
        self.time_since_last_print += elapsed
        if message is not None:
            self.print(message)

    def lap(self, message=None):
        self.stop(message)
        self.start

    def print(self, message=""):
        elapsed = 0
        # if stop() hasn't been called, return current time minus start
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if self.stop_time is not None and self.stop_time > self.start_time:
                elapsed = self.stop_time - self.start_time
        if message != "":
            message = " " + message
        print(f"{self.name}{message}\telapsed {elapsed:0.2f}\tlast {self.time_since_last_print:0.2f} " +
              f"({self.n_since_last_print})\ttotal {self.total_time:0.2f} ({self.n_total})")
        self.n_since_last_print = 0
        self.time_since_last_print = 0


class ConcatTensor():
    def __init__(self, max_len=None):
        """Concatenate tensors along dim 0, preallocate memory if max_len is set.

        Args:
            max_len (int, optional):  if set, preallocate memory for concatenating tensors.
                Total length of dim 0 in filled tensor. Default is None.
        """
        self.idx = 0
        self.max_len = max_len
        self.tensor = [] if self.max_len is None else None

    def _init_shape(self, tensor):
        if self.tensor is None:
            shape = (self.max_len, *tensor.shape[1:])
            self.tensor = torch.empty(shape, dtype=tensor.dtype, layout=tensor.layout,
                                    device=tensor.device, requires_grad=tensor.requires_grad)

    def append(self, tensor: torch.Tensor):
        if self.max_len is None:
            self.tensor.append(tensor)
        else:
            self._init_shape(tensor)
            self.tensor[self.idx:self.idx + tensor.shape[0], ...] = tensor
            self.idx += tensor.shape[0]

    def cat(self):
        if self.max_len is None:
            return torch.cat(self.tensor, dim=0)

        if self.idx < self.max_len:
            warnings.warn(f"ConcatTensor is not filled to {self.max_len} yet!")
        return self.tensor[:self.max_len]


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    dtype = str(dtype)
    if not dtype.startswith("torch."):
        raise ValueError(f"Invalid torch dtype string: {dtype}")
    dtype = dtype.split("torch.")[1]
    return getattr(torch, dtype)


def apply(tensors: Dict[str, torch.Tensor], prefix: str=None, apply_fn: callable=lambda x: x) -> Dict[str, torch.Tensor]:
    return {k if prefix is None else f"{prefix}_{k}": apply_fn(v) for k, v in tensors.items()}


def detach_tensors(tensors: Dict[str, torch.Tensor], to_cpu=True, to_numpy=False):
    def apply_detach(tensor):
        tensor = tensor.detach()
        if to_cpu: tensor = tensor.cpu()
        if to_numpy: tensor = tensor.numpy()
        return tensor
    return apply(tensors, apply_fn=apply_detach)


def to_torch(metrics: Dict[str, torch.Tensor], dtype=torch.float64, prefix=None) -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: torch.tensor(x, dtype=get_dtype(dtype)))


def to_numpy(metrics: Dict[str, torch.Tensor], dtype=torch.float64, prefix=None) -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: x.to(dtype=get_dtype(dtype)).numpy())


def stack_metrics(metrics: Dict[str, torch.Tensor], dim=0, prefix=None) -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: torch.stack(x, dim=dim))


def average_metrics(metrics: Dict[str, torch.Tensor], dim=0, dtype=torch.float64, prefix="avg") -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: torch.mean(x.to(dtype=get_dtype(dtype)), dim=dim))


def variance_metrics(metrics: Dict[str, torch.Tensor], dim=0, dtype=torch.float64, prefix="var") -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: torch.var(x.to(dtype=get_dtype(dtype)), dim=dim))


def last_metrics(metrics: Dict[str, torch.Tensor], dim=0, prefix="last") -> Dict[str, torch.Tensor]:
    return apply(metrics, prefix, lambda x: torch.index_select(x, dim=dim, index=[0]))


def get_available_metrics(metric_dir: Path, include: List[str] = []):
    fields = []
    for file in metric_dir.glob("*"):
        if file.name.endswith("_metrics.pt"):
            if not include or any([x in file.name for x in include]):
                fields.append(file.name.split("_")[:-1] + [file])
    return fields


def combine_dataframes_with_prefix(**prefix_and_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = {}
    for k, v in prefix_and_df.items():
        for column_name, column in v.items():
            combined[f"{k}.{column_name}"] = column
    return pd.DataFrame(combined)


def combine_metrics_into_df(metric_dir: Path, include: List[str] = []):
    metrics = get_available_metrics(metric_dir, include)
    dfs = OrderedDict()
    properties = []
    metric_names = None
    n_examples = None
    for *args, file in metrics:
        metrics_dict = torch.load(file)
        # check that files contain the same metrics
        keys = set(metrics_dict.keys())
        if metric_names is None:
            metric_names = keys
        elif metric_names != keys:
            raise ValueError(f"Metrics differ in {file}: {metric_names}, {keys}")
        # check that every metric has same number of examples
        for k, v in metrics_dict.items():
            if len(v.shape) > 1:
                raise ValueError(f"Metrics should be 1-dimensional: {file}, {k}, {v.shape}")
            if n_examples is None:
                n_examples = v.shape[0]
            elif n_examples != v.shape[0]:
                raise ValueError(f"Metric has wrong size in {file}: {k}, {n_examples}, {v.shape}")
        # make dataframe
        dfs["_".join([str(arg) for arg in args])] = pd.DataFrame(metrics_dict)
        properties.append(args)
    df = combine_dataframes_with_prefix(**dfs)
    properties = list(zip(*properties))
    return df, metric_names, properties


def select_all_replicates(df: pd.DataFrame, *args):
    selected = []
    for key in df.columns:
        if all([k in key for k in args]):
            selected.append(key)
    return df[selected]


def _get_columns_by_name_contains(df: pd.DataFrame, metric_name: str):
    return [x for x in df.columns if metric_name in x]


def _get_columns_by_hparams(df: pd.DataFrame, row: pd.Series):
    run_hash = Path(row['path']).stem
    return [x for x in df.columns if run_hash in x]


def filter_cols_by_hparams(df: pd.DataFrame, hparams: pd.DataFrame):
    columns = []
    for _, row in hparams.iterrows():
        columns = columns + _get_columns_by_hparams(df, row)
    return df[columns]


def filter_cols_by_run(df: pd.DataFrame, row: pd.Series):
    return df[_get_columns_by_hparams(df, row)]


def filter_cols_by_name_contains(df: pd.DataFrame, metric_name: str):
    columns = _get_columns_by_name_contains(df, metric_name)
    return df[columns]


def get_split_mask(df: pd.DataFrame, splits):
    mask = []
    for col in df.columns:
        #TODO hack, need to get model name from col name
        split = filter_cols_by_name_contains(splits, col[:40])
        if len(split.columns) != 1:
            raise ValueError(f"Exactly 1 train/test split required in {split}")
        mask.append(split.to_numpy())
    return np.concatenate(mask, axis=1)


def average_columns(df: pd.DataFrame, reduction="mean", mask=None):
    data = df.to_numpy().astype(np.float32)
    if mask is not None:
        data[mask] = np.nan
    if reduction == "median":
        return np.nanmedian(data, axis=-1)
    elif reduction == "mean":
        return np.nanmean(data, axis=-1)


def print_stats(metrics: Dict):
    for k, v in metrics.items():
        print(k, v.shape, v.dtype, np.min(v), np.mean(v), np.std(v), np.max(v))
