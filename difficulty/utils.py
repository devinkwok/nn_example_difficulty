from collections import OrderedDict
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch

from difficulty.metrics import *


def pointwise_metrics(eval_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    prob = softmax(eval_logits)
    return {
        "acc": zero_one_accuracy(eval_logits, labels),
        "ent": entropy(eval_logits, labels),
        "conf": class_confidence(prob, labels),
        "max-conf": max_confidence(prob),
        "margin": margin(prob, labels),
        "el2n": error_l2_norm(prob, labels),
    }


def train_forget_metrics(eval_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    acc = zero_one_accuracy(eval_logits, labels)
    forget = forgetting_events(acc)
    return {
        "forget": count_forgetting(forget),
        "first-learned": first_learn(forget),
        "first-unforgettable": first_unforgettable(forget),
        "unforgettable": is_unforgettable(forget),
    }


def perturb_forget_metrics(eval_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    acc = zero_one_accuracy(eval_logits, labels)
    forget = perturb_forgetting_events(acc)
    return {
        "forget": count_forgetting(forget),
        "first-forget": perturb_first_forget(forget),
        "unforgettable": is_unforgettable(forget),
    }


def avg_metrics(metrics: Dict):
    return {"avg_" + k: np.mean(v, axis=0) for k, v in metrics.items()}


def last_metrics(metrics: Dict):
    return {"last_" + k: v[-1, ...] for k, v in metrics.items()}


def add_prefix(prefix, metrics: Dict):
    return {prefix + "_" + k: v for k, v in metrics.items()}


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
