import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from difficulty.utils import combine_metrics_into_df, filter_cols_by_hparams, filter_cols_by_name_contains, filter_cols_by_run, get_split_mask, select_all_replicates, average_columns

# args
parser = argparse.ArgumentParser()
parser.add_argument("--metrics_dir", required=True, type=Path)
parser.add_argument("--model_type", required=True, type=str)
args = parser.parse_args()

print("Combining example difficulty metrics with args:", dir(args))

# combine saved metrics into 1 dataframe
df, metrics, (_, models, replicates) = combine_metrics_into_df(args.metrics_dir)

# average metrics by train/test split
hparams = pd.read_csv(args.metrics_dir / "hparams.csv")
runs = hparams.loc[hparams['model_name'] == args.model_type].loc[
                hparams['custom_train_test_split'] == True]
df = filter_cols_by_hparams(df, runs)
# get train/test split info
splits = filter_cols_by_name_contains(df, "is_train")

# do each metric separately
for metric_type in metrics:
    metric_stats = filter_cols_by_name_contains(df, metric_type)
    # get boolean mask for whether each example is in the train or test split
    mask = get_split_mask(metric_stats, splits)
    n_masked = np.count_nonzero(mask, axis=1)
    print(f"{metric_type} train/test split:\tmin={np.min(n_masked)}, avg={np.mean(n_masked)}, max={np.max(n_masked)}, total={mask.shape[1]}")
    # average by mean or median
    for reduction in ["mean", "median"]:
        key = f"replicate-{reduction}.{metric_type}"
        # average over all runs
        df[f"all-{key}"] = average_columns(metric_stats, reduction)
        # average by train or test split only
        for split_type in ["train", "test"]:
            is_train = (split_type == "train")
            df[f"{split_type}-{key}"] = average_columns(
                    metric_stats, reduction, mask=(mask != is_train))

# sanity check that is_train == True for train splits or False for test splits
is_train = filter_cols_by_name_contains(df, "is_train")
is_train_true = filter_cols_by_name_contains(is_train, "train-replicate").to_numpy()
assert np.all(is_train_true == 1)
is_train_false = filter_cols_by_name_contains(is_train, "test-replicate").to_numpy()
assert np.all(is_train_false == 0)

# save combined metrics
save_file = args.metrics_dir / f"{args.model_type}.csv"
print("Combined dataframe:", df)
print(f"Saving stats to {save_file}")
df.to_csv(save_file)
