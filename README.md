Example difficulty
==================

Usage
-----

Experiments require `open_lth` (devinkwok fork).

Run `scripts/train_models.sh` to train `open_lth` models.

Run `scripts/gen-metrics.sh` to generate example difficulty metrics.

Notebooks in `scripts/` contain plots.

Development
-----------

Metrics are implemented in `difficulty/metrics`.

**Functional metrics:** argument names indicate which return values can be used as arguments for other metrics. The dims that metrics operate on are counted from the last dim, allowing the first dims to be arbitrary.

**Accumulator metrics:** these objects are named `Online{Metric}`, and follow the same use pattern: create the object (optionally pass arbitrary metadata that is serializable by `numpy.save`), run `add(data, dim, **metadata)` to update the metric (again, arbitrary metadata can be attached to each update), call `get()` to retrieve current value.

Unit tests for all metrics and some utility functions are under `difficulty/test`. To run tests:

```
python -m unittest discover -s difficulty
```
