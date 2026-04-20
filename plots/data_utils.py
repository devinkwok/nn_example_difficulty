import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata

import sys
if os.getcwd().endswith("plots"):  # this is for notebooks
    sys.path.append("../src")
    OPEN_LTH_PATH = "../../open_lth"
else:  # this is for scripts
    sys.path.append("./src")
    OPEN_LTH_PATH = "./../open_lth"
sys.path.append(OPEN_LTH_PATH)
os.environ["OPEN_LTH_ROOT"] = OPEN_LTH_PATH + "/testing/TESTING/"
os.environ["OPEN_LTH_DATASETS"] = OPEN_LTH_PATH + "/testing/TEST_DATA/"
import api
from difficulty.metrics import precomputed


sns.set_theme(font_scale=1)


MODELS = {
    "cifar_resnet_20": "ResNet-20",
    "cifar_resnet_32": "ResNet deep",
    "cifar_resnet_20_64": "ResNet wide",
    "cifar_vgg_11": "VGG shallow",
    "cifar_vgg_16": "VGG-16",
    "cifar_vgg_16_16": "VGG narrow",
}
DATASETS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cinic10nocifarsubset": "CINIC-10 (excl. CIFAR-10)",
}
HPARAMS = {
    ("training_hparams.lr", 0.01): "LR=0.01",
    ("training_hparams.optimizer_name", "adam"): "Adam",
    ("training_hparams.lr_schedule", "onecycle"): "Cos LR",
}
SCORES = {
    "ddd": "Dichotomous Data Difficulty",
    "carlini_agr": "Ensemble JS-Divergence (agr)",
    "loss": "Loss",
    "acc": "Accuracy",
    "margin": "Area Under Margin",
    "conf": "Confidence",
    "maxconf": "Max Confidence",
    "carlini_conf": "Multi-Model Confidence (conf)",
    "batch_countforget": "Forgetting",
    "batch_firstlearn": "Learning Time",
    "batch_firstunforgettable": "Consistently Learned",
    "batch_unforgettable": "Unforgettable",
    "countforget": "Forgetting (Epoch)",
    "firstlearn": "Learning Time (Epoch)",
    "firstunforgettable": "Consistently Learned (Epoch)",
    "unforgettable": "Unforgettable (Epoch)",
    "grand": "GraNd",
    "el2n": "EL2N",
    "classvog": "VoG",
    "lossvog": "VoG (Cross Entropy)",
    "proto": "Supervised Prototypes",
    "selfproto": "Self-Supervised Prototypes",
    "swav_selfproto": "Self-Supervised Prototypes (SwAV)",
    "pd": "Prediction Depth",
    "carlini_ret": "Holdout Retraining (ret)",
    "carlini_adv": "Adversarial Robustness (adv)",
    "carlini_priv": "Privacy Perserving Training (priv)",
    "memorization": "Memorization",
    "memorization_inception": "Memorization (Inception)",
}
INVERSE = set([
    "conf",
    "maxconf",
    "carlini_conf",
    "batch_unforgettable",
    "unforgettable",
    "carlini_agr",
    "acc",
    "ddd",
    "margin",
    "carlini_ret",
    "carlini_adv",
    "carlini_priv",
])
FUZZ = set([
    "acc",
    "ddd",
    "batch_countforget",
    "batch_firstlearn",
    "batch_firstunforgettable",
    "batch_unforgettable",
    "countforget",
    "firstlearn",
    "firstunforgettable",
    "unforgettable",
    "pd",
])
PRECOMPUTED_SCORES = {
    "carlini_adv": precomputed.load_adv,
    "carlini_ret": precomputed.load_ret,
    "carlini_agr": precomputed.load_agr,
    "carlini_conf": precomputed.load_conf,
    "carlini_priv": precomputed.load_priv,
    "memorization": lambda x: precomputed.load_memorization(x, model=""),
    "memorization_inception": precomputed.load_memorization,
    "swav_selfproto": precomputed.load_self_supervised_prototypes,
}


def flatten_dict(nested_hparams, prefix=None):
    flat_hparams = {}
    for k, v in nested_hparams.items():
        if prefix is not None:
            k = f"{prefix}.{k}"
        if isinstance(v, dict):
            flat_hparams = {**flat_hparams, **flatten_dict(v, prefix=k)}
        else:
            flat_hparams[k] = v
    return flat_hparams


def get_n_trainable_params(hparams):
    model = api.get_model(api.get_model_hparams(hparams))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_df(directory):
    df = []
    for exp in Path(directory).glob("lottery_*"):
        hparams = api.get_hparams_dict(exp)
        hparam_dict = flatten_dict(hparams)
        model = hparam_dict["model_hparams.model_name"]
        dataset = hparam_dict["dataset_hparams.dataset_name"]
        dataset_name = hparam_dict["dataset_hparams.dataset_name"]
        model_name = MODELS[model]
        dataset_name = DATASETS[dataset]
        for (k, v), name in HPARAMS.items():
            if k in hparam_dict and hparam_dict[k] == v:
                model_name = f"{model_name} ({name})"
        df.append({
            "model_key": model,
            "dataset_key": dataset,
            "Model": model_name,
            "Dataset": dataset_name,
            "Task": f"{dataset_name}/{model_name}",
            "Parameters": get_n_trainable_params(hparams),
            "path": exp,
            "key": exp.name,
            "n_replicates": len(list(exp.glob("replicate_*"))),
        })
    return pd.DataFrame(df)


def sanity_checks(path):
    path = Path(path)
    acc_files = defaultdict(list)

    for replicate in path.glob("replicate_*"):
        rep_path = replicate / "level_0" / "main"

        # check that gradmetrics are same as ckptmetrics with same name
        grad_files = {(f.name.split("_")[0], get_ep_it(f)[0]): f for f in rep_path.glob("gradmetrics/*.npz")}
        ckpt_files = {(f.name.split("_")[0], get_ep_it(f)[0]): f for f in rep_path.glob("ckptmetrics/*.npz")}
        for score_and_ep, file in grad_files.items():
            if score_and_ep in ckpt_files:
                grad_score = np.load(file)["arr_0"]
                ckpt_score = np.load(ckpt_files[score_and_ep])["arr_0"]
                if not np.allclose(grad_score, ckpt_score, atol=1e-2, rtol=1e-2):
                    idx = np.where(np.abs(grad_score - ckpt_score) > 1e-2)
                    atol = np.abs(grad_score - ckpt_score) / np.max(np.stack([grad_score, ckpt_score], axis=0), axis=0)
                    print(f"Grad and ckpt scores differ for {file}: n={len(idx[0])}, atol %={np.max(atol)}")

        # check that ensemble_metrics allacc is the average over replicate acc
        for f in rep_path.glob("ckptmetrics/acc_*"):
            acc_files[get_ep_it(f)[0]].append(f)

    for file in path.glob("ensemble_metrics/allacc_*"):
        acc = [np.load(f)["arr_0"] for f in acc_files[get_ep_it(file)[0]]]
        acc = np.mean(np.stack(acc, axis=0), axis=0)
        allacc = np.load(file)["arr_0"]
        if not np.allclose(acc, allacc, atol=1e-2, rtol=1e-2):
            print(f"Ensemble score differ for {file}: n={len(np.where(np.abs(acc - allacc) > 1e-2)[0])}, max={np.max(np.abs(grad_score - ckpt_score))}")


def get_ep_it(path):
    # there are 2 formats: name_epX_itY.npz and name_XepYit.npz
    filename = path.stem
    if "ep" in filename.split("_")[-1]:
        ep, it = filename.split("_")[-1].split("ep")
        ep = int(ep)
        it = int(it.split("it")[0])
    else:
        *_, ep, it = filename.split("_")
        ep = int(ep.split("ep")[1])
        it = int(it.split("it")[1])
    return ep, it


def get_aggregation_method(path):
    if "replicate" not in str(path):
        return "Ensemble"
    key = path.stem.split("_")[0]
    category = path.parent.name
    if category == "pwmetrics" and key == "loss":
        return "One Step"
    if category == "ckptmetrics" or category == "gradmetrics":
        return "One Step"
    if category == "pwmetrics" or category == "batchforget":
        return "Training"


def get_score_key(path):
    splits = path.stem.split("_")
    # skip if not one of the specified scores or if missing ep/it timestamp
    if len(splits) < 2 or splits[0] not in SCORES:
        return None
    # special cases: grandmetrics are same as in ckptmetrics so don't include them
    if path.parent.name == "gradmetrics":
        return None
    # special cases: if parent is batchforget, then change the key
    if path.parent.name == "batchforget":
        return f"batch_{splits[0]}"
    return splits[0]


def get_score_full_name(key, method, epoch, aggregation):
    if method == "One Step" or (method == "Ensemble" and epoch is not None):
        full_name_suffix = f" ({int(epoch)}ep)"
    elif method == "Ensemble":
        full_name_suffix = f" (Ensemble)"
    else:
        full_name_suffix = ""
    return SCORES[key] + ("" if aggregation == "" else f" {aggregation}") + full_name_suffix


def get_score_metadata(score_file, aggregation):
    key = get_score_key(score_file)
    replicate = None
    if "replicate_" in str(score_file):
        replicate = int(str(score_file).split("replicate_")[1].split("/")[0])
    epoch = get_ep_it(score_file)[0]
    method = get_aggregation_method(score_file)
    return {
        "Path": score_file,
        "Replicate": replicate,
        "Epoch": epoch,
        "Method": method,
        "Short Name": SCORES[key],
        "Full Name": get_score_full_name(key, method, epoch, aggregation),
        "Type": aggregation,
        "key": key,
    }


def get_score_rows(score_file):
    rows = []
    key = get_score_key(score_file)
    if key is None:
        return rows

    # some scores have both mean and std over training, others only have the mean
    score_aggregations = [""]
    if len(np.load(score_file).keys()) > 1:
        score_aggregations.append("St. Dev.")

    for aggregation in score_aggregations:
        score_metadata = get_score_metadata(score_file, aggregation)
        rows.append(score_metadata)
    return rows


def get_score_df(model_row, n_replicates=100, get_precomputed=False):
    df = []

    # per replicate scores
    for replicate in range(1, n_replicates + 1):
        replicate_path = model_row["path"] / f"replicate_{replicate}"
        replicate = int(replicate_path.name.split("_")[1])
        for score_file in (replicate_path / "level_0" / "main").glob("*/*.npz"):
            df += get_score_rows(score_file)

    # ensemble scores
    for score_file in (model_row["path"] / "ensemble_metrics").glob("*.npz"):
        df += get_score_rows(score_file)

    df = pd.DataFrame(df)

    # precomputed scores
    if get_precomputed:
        df = pd.concat([df, get_precomputed_metrics(model_row["dataset_key"])])

    return df


def get_precomputed_metrics(dataset):
    df = []
    for key, fn in PRECOMPUTED_SCORES.items():
        try:
            df.append({
                "array": np.array(fn(dataset)),
                "Path": None,
                "Replicate": None,
                "Epoch": None,
                "Method": "Ensemble",
                "Short Name": SCORES[key],
                "Full Name":SCORES[key],
                "Type": "",
                "key": key,
            })
        except Exception as e:
            print(f"Unable to load {key}:", e)
            continue
    return pd.DataFrame(df)


def random_subset_idx(n, total, seed=42):
    return np.random.default_rng(seed).permutation(total)[:n]


def get_scores(df, n_examples=50000, invert=True, fuzz_by=0, randomize_examples_seed=None, total_examples=50000, replace_nan=0):
    new_df = []
    idx = np.arange(n_examples)
    if randomize_examples_seed is not None:
        idx = random_subset_idx(n_examples, total_examples, randomize_examples_seed)

    for _, row in df.iterrows():

        new_row = dict(row)

        if row["Path"] is not None:
            score = np.load(row["Path"])
            if row["Type"] == "":
                arr_key = "arr_0" if len(score.keys()) == 1 else "mean"
            else:
                arr_key = "variance"
            score = score[arr_key]
        else:
            score = new_row["array"]

        # only take a subset of examples
        score = score[idx]

        # convert variance to St. Dev.
        if row["Type"] != "":
            score = np.sqrt(score)

        new_row["is_fuzzed"] = False
        if fuzz_by > 0 and row["key"] in FUZZ:
            score = score.astype(np.float64) + np.random.uniform(-fuzz_by, fuzz_by, size=score.shape)
            new_row["is_fuzzed"] = True

        new_row["is_inverted"] = False
        if invert and row["key"] in INVERSE and row["Type"] == "":
            score = -1 * score
            new_row["is_inverted"] = True

        if replace_nan is not None:
            nans = np.isnan(score)
            nan_idx = np.where(nans)
            if len(nan_idx[0]) > 0:
                print(f"NaNs replaced in {row} at {nan_idx} by {replace_nan}")
            score[np.isnan(score)] = replace_nan

        new_row["array"] = score
        new_df.append(new_row)

    new_df = pd.DataFrame(new_df)
    ensemble_scores = new_df[new_df["Method"] == "Ensemble"]
    per_run_scores = stack_replicates(new_df[new_df["Method"] != "Ensemble"])
    return pd.concat([ensemble_scores, per_run_scores])


GROUPBY_COLS = ["key", "Method", "Type", "Epoch"]


def stack_replicates(df):
    groups = df.groupby(GROUPBY_COLS)
    if len(groups) == 0:
        return df
    sizes = groups.size()
    n = sizes.iloc[0]
    assert np.all(sizes == n)

    new_df = []
    for _, group in groups:
        group = group.sort_values(by="Replicate")
        row = dict(group.iloc[0])
        row["Replicate"] = np.array(group["Replicate"])
        row["array"] = np.stack(group["array"], axis=0)
        new_df.append(row)
    new_df = pd.DataFrame(new_df)

    replicates, arrays = new_df["Replicate"].tolist(), new_df["array"].tolist()
    for reps, arr in zip(replicates, arrays):
        assert np.all(reps == replicates[0])
        assert arr.shape == arrays[0].shape
    return new_df


def aggregate_runs(df, include_std=False, required_runs=None):
    # filter out ensemble scores
    groups = df[df["Method"] != "Ensemble"].groupby(GROUPBY_COLS)

    aggregation_fns = [("", np.mean)]
    if include_std:
        aggregation_fns.append(("St. Dev.", np.std))

    shape = next(iter(groups))[1].iloc[0]["array"].shape
    if required_runs is not None:
        assert shape[0] == required_runs

    new_df = []
    for _, group in groups:
        assert len(group) == 1
        row = group.iloc[0]
        # check that all scores have the same number of runs
        assert row["array"].shape == shape

        for agg_type, fn in aggregation_fns:
            new_row = dict(group.iloc[0])
            new_row["array"] = fn(new_row["array"], axis=0)
            new_row["Method"] = "Ensemble"
            new_row["Type"] = agg_type
            new_row["Full Name"] = get_score_full_name(new_row["key"], "Ensemble", new_row["Epoch"], agg_type)
            new_df.append(new_row)

    return pd.DataFrame(new_df)


def aggregate_steps(df, include_std=False, required_steps=None):
    # only include stepwise scores
    columns = list(GROUPBY_COLS)
    columns.remove("Epoch")
    groups = df[df["Method"] == "One Step"].groupby(columns)

    aggregation_fns = [("", np.mean)]
    if include_std:
        aggregation_fns.append(("St. Dev.", np.std))

    shape = np.stack(next(iter(groups))[1]["array"], axis=0).shape
    if required_steps is not None:
        assert shape[0] == required_steps

    new_df = []
    for _, group in groups:
        arr = np.stack(group["array"], axis=0)
        assert arr.shape == shape

        for agg_type, fn in aggregation_fns:
            new_row = dict(group.iloc[0])
            new_row["array"] = fn(arr, axis=0)
            new_row["Epoch"] = np.array(group["Epoch"])
            new_row["Method"] = "Training"
            new_row["Type"] = agg_type
            new_row["Full Name"] = get_score_full_name(new_row["key"], "Training", max(new_row["Epoch"]), agg_type)
            new_df.append(new_row)
    return pd.DataFrame(new_df)


def get_corr(a, b, corr_fn=spearmanr):
    if len(a.shape) == 1 and len(b.shape) == 1:
        return corr_fn(a, b)[0]
    elif len(a.shape) == 1 and len(b.shape) > 1:
        return np.mean([corr_fn(a, x)[0] for x in b])
    elif len(a.shape) > 1 and len(b.shape) == 1:
        return np.mean([corr_fn(x, b)[0] for x in a])
    else:
        return np.mean([corr_fn(x, y)[0] for x, y in zip(a, b)])


def corr_matrix(values, values_b=None, corr_fn=spearmanr):
    n = len(values)
    if values_b is None:
        corr = np.full([n, n], np.nan)
        for i in range(n):
            for j in range(i+1, n):
                value = get_corr(values[i], values[j], corr_fn=corr_fn)
                corr[i, j] = value
                corr[j, i] = value
    else:
        m = len(values_b)
        corr = np.full([n, m], np.nan)
        for i in range(n):
            for j in range(m):
                value = get_corr(values[i], values_b[j], corr_fn=corr_fn)
                corr[i, j] = value
    return corr


def plot_corr_matrix(matrix, title=None, label_a=None, ticklabels_a=None, label_b=None, ticklabels_b=None, ax=None, show_negative=False):
    ticklabels_b = ticklabels_a if ticklabels_b is None else ticklabels_b
    vmin, vmax = (-0.3, 0.3) if show_negative else (0, 1)
    ax = sns.heatmap(matrix, annot=True, xticklabels=ticklabels_b, yticklabels=ticklabels_a, center=0, vmin=vmin, vmax=vmax, square=True, ax=ax, cbar=False, fmt="0.2f")
    if title is not None:
        ax.set_title(title)
    if label_a is not None:
        ax.set_ylabel(label_a)
        ax.set_xlabel(label_a if label_b is None else label_b)


def get_single_epochs(df, epochs):
    df = df[df["Epoch"].isin(epochs)].sort_values(by="Epoch")
    assert np.all(df["Epoch"] == epochs)
    return np.stack(df["array"], axis=0)


def get_rank_err(values, reference):
    rank = rankdata(values, axis=1)
    ref_rank = rankdata(reference, axis=1)
    err = np.abs(rank - ref_rank) / values.shape[1]
    return err


def filter_df(df, attr_dict):
    for k, v in attr_dict.items():
        df = df[(df[k] == v)]
    return df


def multi_filter_df(df, attr_dicts):
    return pd.concat([filter_df(df, attr) for attr in attr_dicts])


def get_mean_training_score(df, key):
    mean_score = df[(df["Method"] == "Training") & (df["key"] == key)]
    assert len(mean_score) == 1
    return mean_score.iloc[0]["array"]


def subsample_epochs(df, n_samples, randomize=False, random_offset=True):
    if randomize:
        epochs = np.sort(np.random.permutation(np.arange(1, 161))[:n_samples])
    else:
        period = int(160 / n_samples)
        offset = np.random.randint(period) if random_offset else period // 2
        epochs = [i for i in np.arange(1 + offset, 161, period)]
    array = get_single_epochs(df, epochs)
    return array


def get_selected_scores(df, selected_attrs, n_examples=None, strict=True):
    n = df.iloc[0]["array"].shape[-1]
    if n_examples is not None:
        idx = random_subset_idx(n_examples, n)

    for attrs in selected_attrs:
        subset = filter_df(df, attrs)
        if len(subset) != 1:
            if strict:
                raise ValueError(f"{attrs} found the following scores: {subset}")
            else:
                continue

        row = subset.iloc[0]
        score = row["array"]
        if n_examples is not None:
            score = score[:, idx]
        yield row, score


def save_fig(save_path, tag, name=None, figsize=None):
    if figsize is not None:
        fig = plt.gcf()
        fig.set_size_inches(*figsize)
    plt.tight_layout()
    if name is not None:
        name = name.replace("$-1 \\times$", "neg").replace("/", "-").replace("%", "").replace(" ", "_").replace("(", "").replace(")", "")
        save_file = save_path / f"{name}_{tag}.pdf"
        print(f"Saving figure to {save_file}")
        plt.savefig(save_file)
    plt.show()
    plt.clf()