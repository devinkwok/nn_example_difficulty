Example difficulty
==================

Installation
-----

Requirements are in `pyrpoject.toml`.

If computing prediction depth from [Baldock et al. (2021) Deep learning through the lens of example difficulty](https://proceedings.neurips.cc/paper/2021/hash/5a4b25aaed25c2ee1b74de72dc03c14e-Abstract.html), it can be much faster to compute K-nearest-neighbors using the Faiss library.
To enable this, you will need to install [Faiss](https://github.com/facebookresearch/faiss) following the instructions in the linked repository.
Afterwards, set `use_faiss=True` and `device="cuda"` in `prediction_depth` or `representation_metrics`.

Usage
-----

Experiments require a special fork of `open_lth` (included in the repository under `open_lth`).

- Run `open_lth/scripts/difficulty.sh` to train `open_lth` models and generate scores during training.
- Run `open_lth/scripts/collate_metrics.py` to combine results into common directories.
- Run `experiments/trained_ckptmetrics/all_ckptmetrics.sh` to generate example difficulty metrics on completed checkpoints.
- Run `experiments/vit/difficulty_eval.sh` and `experiments/vit/compute_scores_limited.sh` to do inference and generate scores on pretrained ViT.
- Run `plots/plot_bias.py` and `plots/plot_features.py` to run and plot feature selection experiments, as the notebooks are quite slow.

Notebooks in `plots/` contain all plots for the paper.


Development
-----------

Metrics are implemented in `difficulty/metrics`.

**Functional metrics:** argument names indicate which return values can be used as arguments for other metrics. The dims that metrics operate on are counted from the last dim, allowing the first dims to be arbitrary.

**Accumulator metrics:** these objects are named `Online{Metric}`, and follow the same use pattern: create the object (optionally pass arbitrary metadata that is serializable by `numpy.save`), run `add(data, dim, **metadata)` to update the metric (again, arbitrary metadata can be attached to each update), call `get()` to retrieve current value.

Unit tests for all metrics and some utility functions are under `difficulty/test`. To run tests:

```
python -m unittest discover -s difficulty
```
