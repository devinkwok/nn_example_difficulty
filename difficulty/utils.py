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


def save_metrics_dict(metrics: Dict, out_dir: Path, *args):
    out_dir.mkdir(exist_ok=True)
    save_file = "_".join([str(arg) for arg in args]) + "_metrics.pt"
    print(out_dir, save_file)
    torch.save(metrics,  out_dir / save_file)
    return out_dir / save_file


def get_available_metrics(metric_dir: Path):
    fields = []
    for file in metric_dir.glob("*"):
        if file.name.endswith("_metrics.pt"):
            fields.append(file.name.split("_")[:-1] + [file])
    return fields


def combine_dataframes_with_prefix(**prefix_and_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = {}
    for k, v in prefix_and_df.items():
        for column_name, column in v.items():
            combined[f"{k}.{column_name}"] = column
    return pd.DataFrame(combined)


def combine_metrics_into_df(metric_dir: Path):
    metrics = get_available_metrics(metric_dir)
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


def average_columns(df: pd.DataFrame, use_median=False):
    data = df.to_numpy()
    if use_median:
        return np.median(data, axis=-1)
    return np.mean(data, axis=-1)


def print_stats(metrics: Dict):
    for k, v in metrics.items():
        print(k, v.shape, v.dtype, np.min(v), np.mean(v), np.std(v), np.max(v))
