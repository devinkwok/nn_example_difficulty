from itertools import product
from typing import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from difficulty.metrics import rank
from difficulty.utils import select_all_replicates, average_columns


CMAP = plt.get_cmap("plasma")
DIVERGING_CMAP = plt.get_cmap("RdBu")

model_names = OrderedDict()
model_names["ab596c041ffd39d837f0a60d39d86c72"] = "MLP-3"
model_names["06e3ceea2dae7621529556ef969cf803"] = "VGG-16"
model_names["938ede76e304643f5466ed419261dc65"] = "ResNet-20"


def order_metrics(metrics):
    return sorted(metrics, reverse=True)


def order_models(models):
    models = set(models)
    output = OrderedDict()
    for k, v in model_names.items():
        if k in models:
            output[k] = v
    return output


def average_over_replicates(df, model_name, metrics, use_median=False):
    avg_df = pd.DataFrame()
    for metric in metrics:
        df_replicates = select_all_replicates(df, metric, model_name)
        avg_df[metric] = average_columns(df_replicates, use_median=use_median)
    return avg_df


def make_ranks(df):
    rank_df = {}
    for name, column in df.iteritems():
        rank_df[name] = rank(column)
    return pd.DataFrame(rank_df)


def plot_heatmap_tiles(ax, array, x_labels, y_labels, title="", plot_diagonal=True):
    if not plot_diagonal:
        for i in range(min(array.shape)):
            array[i, i] = 0.
    image = ax.imshow(
        array,
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1),
        cmap=DIVERGING_CMAP,
    )
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    #TODO color bar legend
    ax.figure.colorbar(image, ax=ax)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    for x, y in product(range(len(x_labels)), range(len(y_labels))):
        if plot_diagonal or x != y:
            value = array[y, x]
            text_color = "black" if abs(value) < 0.5 else "white"
            ax.text(x, y, f"{value:0.2f}", ha="center", va="center", color=text_color)


def test_plot_heatmap_tiles():
    ax = plt.subplot()
    values = np.arange(12).reshape(3,4) / 6 - 1
    print(values)
    plot_heatmap_tiles(ax, values, 4 - np.arange(4), np.arange(3))
