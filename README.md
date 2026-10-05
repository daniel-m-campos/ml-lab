# ml-lab

A local-first record of experimentation on frozen datasets with a fixed row order: time series when there is a clock, any tabular data once shuffled into one. Declare pipelines and evaluations in Python; `lab run` computes only what the log lacks, refuses the common leaks before recording anything, and writes every result as an event naming the code, data and environment behind it. Read with SQL.

- **Memo.** A fit, prediction or score is reused when the log holds one for the same data, declaration, code and environment. A rerun of an unchanged tree writes nothing.
- **Identity from content.** A dataset id is its rows, a pipeline id its declaration. Edit code and the affected fits rerun; edit a comment and nothing does. Names are labels.
- **Guards that refuse.** A function sees rows only up to where it may look; unknown labels are NaN; a fold that trains on labels revealed after its test range starts is refused; every prediction is recomputed on a shorter prefix and must agree.
- **Provenance.** A score names its run, predictions and fits; a run records the commit, dirty diff, host and the tool's commit.

## Install

```
git clone git@github.com:daniel-m-campos/ml-lab.git
uv pip install -e ml-lab                 # Python 3.12, numpy, polars
```

Development: `uv pip install -e ".[dev]"` and `pytest`.

## The smallest project

`examples/minimal/project.py`, one file: synthetic rows, one ridge pipeline in two variants, one evaluation.

```
export ML_LAB_ROOT=$PWD/.ml-lab-minimal ML_LAB_ACTOR=$USER
lab ingest examples/minimal/project.py                  # prints the dataset id
lab run examples/minimal/project.py                     # fits 6, predictions 6, scores 2; a rerun writes nothing
sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT name, metric, rank, round(fold_mean, 4) AS fold_mean, n_folds FROM board WHERE source = 'toy'"
```

```
┌──────────────┬────────┬──────┬───────────┬─────────┐
│     name     │ metric │ rank │ fold_mean │ n_folds │
├──────────────┼────────┼──────┼───────────┼─────────┤
│ ridge        │ mse    │ 1    │ 0.9933    │ 3       │
│ ridge_shrunk │ mse    │ 2    │ 0.9941    │ 3       │
└──────────────┴────────┴──────┴───────────┴─────────┘
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
        targets=("y",),
    )


@dataclasses.dataclass(frozen=True)
class RidgeParams:
    alpha: float = 1.0


def ridge_fit(dataset: Dataset, train: Segments, params: RidgeParams) -> np.ndarray:
    x = dataset.column("x", train)
    y = dataset.column("y", train)
    return np.array([x @ y / (x @ x + params.alpha)])


def ridge_predict(model: np.ndarray, dataset: Dataset, rows: Range) -> np.ndarray:
    return model[0] * dataset.column("x", rows)


def ridge_save(model: np.ndarray) -> bytes:
    return formats.arrays_save({"slope": model})


def ridge_load(payload: bytes) -> np.ndarray:
    return formats.arrays_load(payload)["slope"]


def squared_error(
    pred: np.ndarray, dataset: Dataset, rows: Range, params
) -> np.ndarray:
    return (pred - dataset.column("y", rows)) ** 2


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
    format=formats.Format.ARROW_ARRAYS,
    params=RidgeParams(),
)
pipelines = [ridge, ridge.with_params(alpha=100.0).named("ridge_shrunk")]


def evaluations(dataset: str) -> list[Evaluation]:
    return [
        Evaluation(
            name="validation",
            dataset_id=dataset,
            split=WalkForward(first_cutoff_rows=2000, step_rows=1000, test_rows=1000),
            scorer=mse,
        )
    ]
```

- **`Dataset(columns, ts=None)`**: equal-length 1-D numpy columns in a fixed order, optional sorted `ts`; `ml_lab.dataset.from_frame(frame)` builds one from polars. Functions read it with `dataset.column(name, rows)` and `dataset.matrix(cols, rows)`. `record` stores the rows and returns an id that is a hash of them; `targets` names the labels the tool hides, and an empty `targets` is refused. Extra `lab ingest` arguments reach `dataset(ledger, *args)` as strings.
- **`fit(dataset, train, params)`**: `train` is the train row segments. A fit sees targets and features as NaN outside its train segments, so it cannot read its validation block at all. Returns any object.
- **`predict(model, dataset, rows)`**: one row per row of the range, `(rows,)` or `(rows, k)`. Unknown labels inside the range are NaN. The tool recomputes the range on a shorter prefix and refuses the function if earlier rows changed: a feature reading later rows or a non-deterministic model is caught with the first moved row.
- **`save(model)` and `load(payload)`**: bytes in the pipeline's `format`, one of `formats.KNOWN`. `arrow-arrays` and `zip` open without Python; for sklearn, `formats.pickle_save` and `formats.pickle_load` with `format=formats.Format.PICKLE`, marked non-portable. A fit is recorded only after the reloaded model predicts its first range as the fitted one did.
- **`Scorer(series, metrics, directions)`**: `series(pred, dataset, rows, scorer_params)` returns the series stored for later paired tests. A series is `(rows,)` or `(rows, c)` with one row per scoring unit, a prediction or a day; `Scorer(columns=...)` names the columns. `metrics(series)` reduces it per fold (`fold_mean`, `fold_std`) and over the concatenated folds (`pooled`). Only `series` enters the evaluation id; `metrics` and `directions` are code under the memo. Keep the scorer in its own module: a fit's memo covers every module its functions import.
- **`Pipeline`**: functions, `format` and `params`, a frozen dataclass in a module `fit` imports, so an edited default refits. Its id is the declaration; `name` is a label. A lambda, closure or method is refused in a `Pipeline` slot or as `Scorer.series`; `metrics` may be a lambda. `with_params` and `named` make variants.
- **`Evaluation(dataset_id, split, scorer)`**: `evaluations(dataset_id)` returns the list every pipeline is scored under. A split is a frozen dataclass whose `folds(dataset)` returns `Fold(label, train, test)` items, so the ingest order is the split's design: shuffled rows make `BlockedKFold` a random k-fold, rows reordered by `ml_lab.dataset.stratified_order(y)` a stratified one. The folds are recorded with the evaluation, and a split whose code moved them is refused.

## Evaluations and splits

An evaluation is a split and a scorer over one dataset. The split turns the dataset into folds, and a fold is train row segments and one test range. The tool fits on the train segments, predicts the test range with that fit and scores it against the real labels, so every scored prediction is out of sample for its fit; a fold whose test range overlaps its train segments is refused. The example's `WalkForward(first_cutoff_rows=2000, step_rows=1000, test_rows=1000)` gives:

| fold | trains on | test range |
|---|---|---|
| row 2000 | rows 0 to 2000 | rows 2000 to 3000 |
| row 3000 | rows 0 to 3000 | rows 3000 to 4000 |
| row 4000 | rows 0 to 4000 | rows 4000 to 5000 |

`BlockedKFold(n_splits=5)` validates each of five blocks with the other four as train, `Holdout(train_size=0.8)` is one fold, `WalkForward(max_train_rows=1000)` trains on a rolling window, and `embargo_rows` drops rows between the two ranges. `group="g"` on `BlockedKFold` or `Holdout` moves each cut to the first row of its group, so no group is split; a group that recurs after another is refused. Decay is another evaluation: `WalkForward(horizon=3)` scores the third block after each cutoff and shares every fit with `horizon=1` through the memo, so it costs predictions only.

Growing it:

- `Pipeline(features=f)` with `f(dataset) -> {name: array}`: columns computed once per dataset, shared by every pipeline declaring `f`, read by the names in `dataset.computed_columns`, probed on shorter prefixes.
- `Pipeline(postprocess=clip, postprocess_config=Clip(at=3.0))`: a stateless stage after `predict`, called as `clip(pred, dataset, rows, config)`; its config enters the prediction id, not the fit's.
- `Pipeline(members=(a, b))`: a blend whose `predict` takes the members' predictions as a fourth argument.
- A test set: `record(..., sealed_from=4000)`, a row or a date, seals the tail. An evaluation whose folds validate past it is refused unless it is `Evaluation(sealed=True)`, so validate on `BlockedKFold(n_splits=4, train_size=0.8)` or a split that ends before it. A sealed evaluation scores each pipeline once; the pick is the pipeline list committed in `test.py` before the run:

```python
from experiment import mse, pipelines as candidates

from ml_lab.experiment import Evaluation
from ml_lab.splits import Holdout

pipelines = [p for p in candidates if p.name == "ridge_shrunk"]


def evaluations(dataset: str) -> list[Evaluation]:
    return [Evaluation(dataset, Holdout(train_size=0.8), mse, sealed=True, name="test")]
```

## Time series

`Dataset(columns, ts)` adds a clock: sorted timestamps, several rows to a timestamp for a cross-section. Every row-based cut lands on the first row of its timestamp. `CalendarWalkForward(first_cutoff="2021-05-04")` cuts every month, or every day with rows with `unit="day"`, and scores the next `test_units` units after each cutoff; `horizon=3` is another evaluation that shares every fit. A label known after its row's timestamp declares a reveal lag, `record(..., reveal={"y": timedelta(days=1)})` or an integer count of later dates with rows (`1` is the next trading day's open); labels not yet known are NaN to `predict`. `embargo_timestamps` (or `embargo_rows`) drops the train rows just before each cutoff. `ml_lab.panel.Panel(dataset, key)` grids a cross-section by (time, key) inside a features function, with `pivot` and `unpivot`.

- **"folds train on labels revealed after their test range starts"**: the embargo is shorter than the reveal lag; the message says how many more timestamps to purge.
- **"a month is whole or dropped"**: the last month runs past the data; set `end=` or use `unit="day"`.

## Run and read

```
export ML_LAB_ROOT=$PWD/ledger ML_LAB_ACTOR=daniel   # ML_LAB_ACTOR is required: every write names its actor
lab ingest project/dataset.py 20                     # calls dataset(ledger, "20")
lab run project/experiment.py                        # the newest dataset when the ledger holds one source; otherwise --dataset <source or id prefix>
lab run project/experiment.py ideas/agent7.py        # pipelines from both files, the evaluations from one
lab run project/experiment.py --dry-run              # every id computed, nothing written: what would fit and why
```

A pipeline that raises is recorded as a failure and the rest continue; `lab run` exits 1 naming it. Views over the log, in `ml_lab.sqlite`, for `sqlite3` or DuckDB:

```sql
SELECT name, actor, metric, direction, rank, fold_mean, fold_std, n_folds, pooled FROM board
WHERE source = 'toy' AND evaluation_name = 'validation' ORDER BY metric, rank;

SELECT name, reference_name, delta_mean, delta_std, t, wins, n_folds FROM head_to_head
WHERE source = 'toy' AND reference_name = 'ridge' AND metric = 'mse';
```

`board` holds the latest score per evaluation, pipeline and metric, whoever recorded it; `rank` is 1 for the best by `direction`. `head_to_head` pairs pipelines fold by fold: `delta_mean` is name minus reference, `t` is fold-paired, `wins` follow the direction, and `pair_fold` holds the per-fold rows. At one fold, `ml_lab.paired` pairs the stored series row by row:

```python
from ml_lab import Ledger, paired

ledger = Ledger(".ml-lab-minimal")
score = {r["name"]: r["score"] for r in ledger.sql("SELECT name, score FROM board")}
paired(ledger, score["ridge_shrunk"], score["ridge"], "0", block_rows=250)
```

Predictions are Parquet files under `blobs/sha256/`, named by `event_prediction.blob`. The event tables, views, blob formats, declined designs and what is deferred: [docs/spec.md](docs/spec.md).

## Trust boundary

The guards see what a function receives. A function that reads a file, an environment variable, module globals set elsewhere, `.base` or `as_strided` on a view, or the feature store's path, or that caches across pipelines in a module global, or whose output depends on global random state, is outside the guards. The process is the trust boundary.

## Refusals you will meet

- **"read past row N"**: a function's output changed when recomputed on a shorter prefix; the first moved row and both values are named.
- **"the model loaded from its saved bytes predicts differently"**: `save` loses what `predict` reads.
- **"is defined in ..., which the fit function does not import"**: a params class outside the memo's reach.
- **"is not a module-level function of ..."**: a lambda, closure or method in a slot.
- **"yields other folds than evaluation ... recorded"**: a split's code moved its folds; rename it or change a field.
