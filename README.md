Example difficulty
==================

Usage
-----

Requires `open_lth` (devinkwok fork) as submodule, run:

```
git submodule add git@github.com:devinkwok/open_lth.git
git submodule init && git submodule update
```

Run `scripts/train_models.sh` to train `open_lth` models.

Run `scripts/gen-metrics.sh` to generate example difficulty metrics.

Notebooks in `scripts/` contain plots.

Development
-----------

Metrics are implemented in a functional style in `difficulty/metrics`. Argument names indicate which return values can be used as arguments for other metrics. The dims that metrics operate on are counted from the last dim, allowing the first dims to be arbitrary.

Unit tests for all metrics and some utility functions are under `difficulty/test`. To run tests:

```
python -m unittest discover difficulty/test
```
