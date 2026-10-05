# ml-lab

A local-first record of experimentation on frozen datasets with a fixed row order: time series when there is a clock, any tabular data once shuffled into one. Declare pipelines and evaluations in Python; `lab run` computes only what the log lacks, refuses the common leaks before recording anything, and writes every result as an event naming the code, data and environment behind it. Read with SQL.

- **Memo.** A fit, prediction or score is reused when the log holds one for the same data, declaration, code and environment. A rerun of an unchanged tree writes nothing.
- **Identity from content.** A dataset id is its rows, a pipeline id its declaration, a fit id the code keys (syntax trees, so comments do not count) of every module its functions import plus the environment lock. Names are labels.
- **Guards that refuse.** A function sees rows only up to where it may look; unknown labels are NaN; a fold that trains on labels revealed after its window starts is refused; every window is recomputed on a shorter prefix and must agree.
- **Provenance.** A score names its run, predictions and fits; a run records the commit, dirty diff, host and the tool's commit.
- **One ledger, many writers.** People and agents append to one SQLite file under an actor name.

## Install

```
git clone git@github.com:daniel-m-campos/ml-lab.git
uv pip install -e ml-lab                 # Python 3.12, numpy, polars
```

Editable, because the environment lock hashes the tool's code: a `git pull` that changes a fit refits what it touched, a docs pull refits nothing. A ledger survives a pull unless the schema version moves, which `lab` refuses with the instruction to delete and rerun. Feedback is a GitHub issue with the `friction` label and a minimal repro; `SELECT editable FROM raw_run` names the tool commit a run used. Development: `uv pip install -e ".[dev]"` and `pytest`.

## The smallest project

`examples/minimal/project.py`, one file: synthetic rows, one ridge pipeline in two variants, one evaluation.

```
export ML_LAB_ROOT=$PWD/.ml-lab-minimal
lab ingest examples/minimal/project.py                  # prints the dataset id
lab run examples/minimal/project.py                     # fits 6, predictions 6, scores 2; a rerun writes nothing
sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT name, metric, fold_mean, fold_std, folds FROM board"
```

```
┌──────────────┬────────┬───────────┬──────────┬───────┐
│     name     │ metric │ fold_mean │ fold_std │ folds │
├──────────────┼────────┼───────────┼──────────┼───────┤
│ ridge_shrunk │ mse    │ 0.9941    │ 0.0607   │ 3     │
│ ridge        │ mse    │ 0.9933    │ 0.0591   │ 3     │
└──────────────┴────────┴───────────┴──────────┴───────┘
```

```python
from __future__ import annotations

import dataclasses

import numpy as np

from ml_lab import formats
from ml_lab.dataset import Dataset, Range, Segments, record
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Ledger
from ml_lab.splits import WalkForward


def dataset(ledger: Ledger, rows: str = "5000") -> str:
    """Rows of one feature and one noisy target, in a fixed order; prints the id."""
    random = np.random.default_rng(0)
    x = random.normal(size=int(rows))
    y = 0.5 * x + random.normal(size=int(rows))
    return record(
        ledger,
        Dataset({"x": x, "y": y}),
        source="toy",
        params={"rows": rows},
        filters=(),
        targets=("y",),
    )


@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0


def ridge_fit(dataset: Dataset, train: Segments, config: RidgeConfig) -> np.ndarray:
    x = dataset.column("x", train)
    y = dataset.column("y", train)
    return np.array([x @ y / (x @ x + config.alpha)])


def ridge_predict(model: np.ndarray, dataset: Dataset, rng: Range) -> np.ndarray:
    return model[0] * dataset.column("x", rng)


def ridge_save(model: np.ndarray) -> bytes:
    return formats.arrays_save({"slope": model})


def ridge_load(payload: bytes) -> np.ndarray:
    return formats.arrays_load(payload)["slope"]


def squared_error(pred: np.ndarray, dataset: Dataset, rng: Range, config) -> np.ndarray:
    return (pred - dataset.column("y", rng)) ** 2


mse = Scorer(
    squared_error,
    metrics=lambda series: {"mse": float(series.mean())},
    directions={"mse": "min"},
)


ridge = Pipeline(
    name="ridge",
    fit=ridge_fit,
    predict=ridge_predict,
    save=ridge_save,
    load=ridge_load,
    config=RidgeConfig(),
    format=formats.Format.ARROW_ARRAYS,
)
pipelines = [ridge, ridge.with_config(alpha=100.0).named("ridge_shrunk")]


def evaluations(dataset: str) -> list[Evaluation]:
    return [
        Evaluation(
            name="validation",
            dataset=dataset,
            split=WalkForward(first_cutoff_rows=2000, step_rows=1000, window_rows=1000),
            scorer=mse,
        )
    ]
```

- **`Dataset`**: equal-length numpy columns in a fixed order, optional sorted `ts`. `record` stores the rows and returns an id that is a hash of them; `targets` names the labels the tool will hide.
- **`fit(dataset, train, config)`**: `train` is the train row segments; the dataset ends at the last train row and targets are NaN outside the segments, so a fit cannot read its window or a k-fold validation block. Returns any object.
- **`predict(model, dataset, rng)`**: one row per window row, `(rows,)` or `(rows, k)`. Unknown labels inside the window are NaN. The tool recomputes the window on a shorter prefix and refuses the function if earlier rows changed: a feature reading later rows or a non-deterministic model is caught with the first moved row.
- **`save(model)` and `load(payload)`**: bytes in the pipeline's `format`, one of `formats.KNOWN` (`arrow-arrays` and `zip` open without Python, `pickle` is marked non-portable). A fit is recorded only after the reloaded model predicts the first window as the fitted one did.
- **`Scorer(score, metrics, directions)`**: `score(pred, dataset, rng, config)` returns a series, one row per scoring unit; `metrics(series)` reduces it, per fold (`fold_mean`, `fold_std`) and over the concatenated folds (`value`). The series is stored for later paired tests. Only `score` enters the evaluation id; `metrics` and `directions` are code under the memo.
- **`Pipeline`**: functions, config, format. Its id is the declaration; `name` is a label. Any module-level function fills a slot; a lambda, closure or method is refused. A config is a frozen dataclass in a module `fit` imports, so an edited default refits. `with_config` and `named` make variants.
- **`Evaluation`**: dataset, split, scorer, config; `evaluations(dataset_id)` returns the list every pipeline is scored under. A split is a frozen dataclass with `folds(dataset)` returning contiguous row ranges, so the ingest order is the split's design: shuffled rows make `BlockedKFold` a random k-fold, interleaved by class a stratified one. The folds are recorded with the evaluation and a split whose code moved them is refused.

Growing it:

- `features(dataset) -> {name: array}`: columns computed once per dataset, shared by every pipeline declaring the function, read via `dataset.feature_columns`; probed on five prefixes.
- `Pipeline(postprocess=clip, postprocess_config=Clip(at=3.0))`: a stateless stage after `predict`, called as `clip(pred, dataset, rng, config)`; its config enters the prediction id, not the fit's.
- `Pipeline(members=(a, b))`: a blend whose `fit` and `predict` take the members' predictions as a fourth argument.
- `Dataset(columns, ts)`: a clock adds `CalendarWalkForward` (cutoffs in months or trading days, named horizons, an embargo in timestamps) and reveal lags, `record(..., reveal={"y": timedelta(days=1)})` or an integer count of trading dates.
- A test set is a second `Evaluation` on `Holdout(train_fraction)` in its own `test.py`, run once after the pick is written; `BlockedKFold(k, train_fraction)` validates on the same cut. The `ml-lab-exp` studies are the template.

## Run and read

```
export ML_LAB_ROOT=$PWD/ledger ML_LAB_ACTOR=daniel
lab ingest project/dataset.py 20
lab run project/experiment.py                          # newest dataset of its source; --dataset <source or id prefix>
lab run project/experiment.py ideas/agent7.py          # pipelines from both files, the evaluations from one
lab run project/experiment.py --dry-run                # every id computed, nothing written: what would fit and why
```

A pipeline that raises is recorded as a failure and the rest continue; `lab run` exits 1 naming it. Views over the log, in `ml_lab.sqlite`, for `sqlite3` or DuckDB:

```sql
SELECT name, metric, fold_mean, fold_std, folds, value FROM board
WHERE evaluation_name = 'validation' AND window = '1' ORDER BY metric, fold_mean DESC;

SELECT name, mean_delta, delta_std, t, wins, folds FROM head_to_head
WHERE reference_name = 'ridge_3m' AND evaluation_name = 'validation' AND window = '1' AND metric = 'pnl';

SELECT f.fold, f.label, f.window, f.metric, f.value FROM score_latest l
JOIN score_fold f ON f.score = l.score WHERE l.name = 'ridge_3m';

SELECT p.name, x.error, x.run FROM raw_failure x JOIN raw_pipeline p ON p.id = x.pipeline;
```

`board` and `head_to_head` are for reading; `raw_<event>` is one event type as columns; `score_fold`, `score_aggregate`, `score_latest`, `score_fit` are the building blocks. The event and view tables, blob formats, declined designs and what is deferred: [docs/spec.md](docs/spec.md).

## Refusals you will meet

- **"fold windows train on labels revealed after they start"**: the embargo is shorter than the reveal lag; the message says how many more timestamps to drop.
- **"read past row N"**: a function's output changed when recomputed on a shorter prefix; the first moved row and both values are named.
- **"source changed during the run"**: a module a function imports was edited mid-run; rerun.
- **"the model loaded from its saved bytes predicts differently"**: `save` loses what `predict` reads.
- **"is defined in ..., which the fit function does not import"**: a config class outside the memo's reach.
- **"is not a module-level function of ..."**: a lambda, closure or method in a slot.
- **"yields other folds than evaluation ... recorded"**: a split's code moved its folds; rename it or change a field.
- **"ledger schema N, this build is M"**: delete the root and rerun.

Layout: `src/ml_lab/ledger.py` (events, views, blobs), `experiment.py` (Pipeline, Evaluation, Scorer), `splits.py`, `runs.py` (memo, guards, probes), `dataset.py` (Dataset, record, load), `formats.py`, `identity.py` (hashes, code keys, import closures, the lock), `panel.py` ((time, key) grids), `cli.py`. Background: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md).
