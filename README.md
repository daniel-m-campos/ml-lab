# ml-lab

A local-first record of experimentation on frozen datasets whose rows are in a fixed order: time series when there is a clock, any tabular data once shuffled into one. For one person or a team of agents working the same data. You declare pipelines and evaluations in plain Python; `lab run` fits, predicts and scores only what the log does not already hold, refuses the common leaks before they are recorded, and writes every result as an event that names the code, data and environment behind it. You read the results with SQL.

It exists because experiment tracking tools record what you tell them, and the things that go wrong (a feature that reads the future, a label known a day later than its row, a k-fold fit that saw its validation labels, a fit rerun under edited code, two "identical" runs with different numbers) are not things anyone tells them. Here identity is computed from content, the guards run on every write, and the reads are tables.

## What you get

- **A memo over everything.** A fit is reused when one exists for the same dataset, declaration, train range, code and environment; so are predictions and scores. A rerun of an unchanged tree writes nothing and says what it reused. Adding one pipeline costs only that pipeline.
- **Identity from content.** A dataset id is its rows; a pipeline id is its declaration; a fit id covers the code keys (syntax trees, so comments and formatting do not count) of every module its functions import, plus the lock of every distribution they reach. Names are labels and never enter an id.
- **Leak guards that refuse, not warn.** Every function sees a prefix of the data ending where it may look. Labels not yet known at a window's start are NaN inside it. A fold that trains on labels revealed after its window starts is refused before anything runs, with the embargo that clears it. Every computed window is recomputed on a shorter prefix and must agree, so a function that reads later rows is refused with the first row that moved.
- **Provenance per number.** A score names its run, predictions and fits; a run records the commit, the dirty diff, the host and the tool's own commit; a fit records its import shas and environment lock.
- **One ledger, many writers.** Agents and people append to the same SQLite file under an actor name. A research repo with one directory per study and one shared ledger is the pattern the `ml-lab-exp` studies use.

## Install

```
git clone git@github.com:daniel-m-campos/ml-lab.git
uv pip install -e ml-lab                 # editable, into the project's venv; Python 3.12, numpy and polars
```

Editable is the point: the environment lock hashes the tool's code, so a `git pull` that changes how a fit is computed refits what it touched and a pull that changes docs refits nothing. A non-editable install carries only the version string and would reuse fits across tool changes. Upgrading is `git pull`; a ledger survives it unless the schema version moves, which `lab` refuses loudly with the instruction to delete and rerun. Every run records the tool's commit and dirty diff (`SELECT editable FROM raw_run`), so an issue can name the exact tool it hit. Feedback goes in as a GitHub issue with the `friction` label, with a minimal repro; fixes land by pull request and bump the patch version.

For development: `uv venv .venv && uv pip install -e ".[dev]"` then `.venv/bin/python -m pytest`. The suite is the ledger story on synthetic data, `tests/test_ledger_story.py`, and every refusal the tool makes is pinned there.

## A project, from the smallest one

`examples/minimal/project.py` is a whole project in one file: synthetic rows, one ridge pipeline in two variants, one evaluation. Run it before reading it:

```
export ML_LAB_ROOT=$PWD/.ml-lab-minimal
lab ingest examples/minimal/project.py                  # 289160eafabf965b
lab run examples/minimal/project.py
sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT name, metric, fold_mean, fold_std, folds FROM board"
```

```
fit ridge row 2000 0.0s: no earlier fit of this pipeline and label
predictions ridge test rows 2000:3000
...
score ridge test mse=0.9933
run 01M4692Z3DXSDNNRQ5HC8JZQQ3: fits 6, predictions 6, scores 2
┌──────────────┬────────┬───────────┬──────────┬───────┐
│     name     │ metric │ fold_mean │ fold_std │ folds │
├──────────────┼────────┼───────────┼──────────┼───────┤
│ ridge_shrunk │ mse    │ 0.9941    │ 0.0607   │ 3     │
│ ridge        │ mse    │ 0.9933    │ 0.0591   │ 3     │
└──────────────┴────────┴───────────┴──────────┴───────┘
```

Run it a second time and it prints `nothing written`: every fit, prediction and score is already in the log. The file, top to bottom:

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

What each piece is, in the order the file introduces it:

**The dataset** is a `Dataset`, a dict of equal-length numpy columns in a fixed order, with an optional sorted `ts` clock (none here). `record` stores the rows as Parquet and returns the dataset id, which is a hash of the rows themselves, so a loader fix that changes a value is a new dataset and re-ingesting identical rows writes nothing. `targets` names the label columns, because the tool has to know which columns to hide from a function that must not see them. `lab ingest` calls `dataset(ledger, *args)` and prints the id.

**`fit`** receives the dataset, its train rows as `Segments` (a tuple of row ranges), and the config. It reads columns with `dataset.column(name, rows)`; the dataset it sees ends at the last train row, and a read past that raises, so a fit cannot touch its scoring window by accident. Its targets are NaN outside the train segments, so a k-fold fit with a validation block in the middle cannot read that block's labels either. It returns the model as any Python object.

**`predict`** receives the model, the dataset and the window's row range, and returns one row per window row: a `(rows,)` array, or `(rows, k)` for a multi-class model. The dataset it sees ends at the window's end, and inside the window every label not yet known is NaN. Before the predictions are written, the tool recomputes them on a shorter prefix of the window and refuses `predict` if the earlier rows changed, which is how a feature that reads later rows, or a non-deterministic model, is caught with the first row that moved.

**`save` and `load`** turn the model into bytes and back. The pipeline's `format`, one of `formats.KNOWN`, names what `save` writes; `arrow-arrays` (a dict of arrays) and `zip` open without Python, `pickle` is admitted for scikit-learn and marked non-portable. A fit is recorded only after the reloaded model predicts the first window exactly as the fitted one did, so what the log holds is what the bytes can reproduce.

**The scorer** is a `Scorer`. Its `score` function receives the predictions, the dataset, the window range and the evaluation's config, and returns a series with one row per scoring unit (here a squared error per row; for a cross-sectional metric, one row per day). `metrics` reduces a series to named numbers and runs twice: per fold, which gives `fold_mean` and `fold_std`, and over the concatenated folds, which gives `value`. The series is stored, so a paired test or a bootstrap can be run later from the log alone. `directions` says which way is better; `head_to_head` reads it to count wins. Only `score` enters the evaluation id, so an edit to `metrics` or `directions` is a new score under the same evaluation.

**`Pipeline`** binds the functions, a config and the model format. Its id is a hash of the declaration (the functions' dotted paths, the format and the config's values), and `name` is a label outside the hash, so `ridge_shrunk` is one id under one name. Any module-level function fills a slot; a lambda, a closure or a method has no dotted path and is refused. The config is a frozen dataclass defined in a module `fit` imports, so an edited default is caught by the fit memo rather than silently reusing an old fit. `with_config` and `named` make variants; a variant that differs only in `postprocess` shares every fit with its parent.

**`Evaluation`** is a dataset, a split, a scorer and a config, and `evaluations(dataset_id)` returns the list `lab run` scores every pipeline under. `WalkForward` here cuts at rows 2000, 3000 and 4000, trains on everything before each cut and scores the thousand rows after it. A split is any frozen dataclass with `folds(dataset)`, and folds are contiguous ranges in the stored row order, so the order chosen at ingest is the split's design: shuffle in a filter function and `BlockedKFold` is a random k-fold; interleave by class and it is stratified. The folds are recorded with the evaluation and checked on every run, so a split whose code changed under the same declaration is refused rather than scored against stale folds.

**Why a rerun writes nothing.** A fit's id covers the dataset, the fit declaration, the train range, the code keys of every module the functions import (syntax trees, so comments and formatting do not count) and the environment lock of every distribution they reach. Edit `ridge_fit` and both pipelines refit; edit this README and nothing does. Predictions are keyed by fit and range, scores by their prediction ids. `lab run --dry-run` prints what would be computed and, per stale fit, which files moved.

### Growing it

- **Variants and sweeps**: `ridge.with_config(alpha=a).named(f"ridge_{a}")` in a list; a scoring sweep is a list of `dataclasses.replace(evaluation, config=..., name=...)`.
- **A `features` function**: `features(dataset) -> {name: array}` computes columns once per dataset, shared by every pipeline that declares the function, read by the names in `dataset.feature_columns`. It is probed on five prefixes, so a feature that reads its row's future is refused at first computation.
- **`postprocess`**: `Pipeline(postprocess=clip, postprocess_config=Clip(at=3.0))` adds the cheap stateless stage after `predict` (clip, rank, neutralise), called as `clip(predictions, dataset, rng, config)`; its config enters the prediction's identity and not the fit's.
- **Blends**: `Pipeline(members=(a, b), ...)`; `fit` and `predict` take the members' predictions as a fourth argument, and the members' fits are memoized on their own.
- **A clock**: `Dataset(columns, ts)` with a sorted datetime array. It adds `CalendarWalkForward` (cutoffs every N months or trading days, named horizons, an embargo in whole timestamps), reveal lags on targets (`record(..., reveal={"y": timedelta(days=1)})`, or an integer count of trading dates), and the refusal of any fold that trains on labels revealed after its window starts.
- **A test set**: a second `Evaluation` on `Holdout(train_fraction)` in its own module, `test.py`, run once after the pick is written down; `BlockedKFold(k, train_fraction)` validates on the same cut.
- **The real-world template**: `examples/optiver/` adds a clock, three horizons, a per-stock taker simulation as the scorer and two model families (ridge, bonsai). `examples/optiver/launch.sh 20` runs it when the Kaggle "Trading at the Close" `train.csv` sits under `data/optiver/optiver-trading-at-the-close/`.

## Run

```
export ML_LAB_ROOT=$PWD/ledger ML_LAB_ACTOR=daniel
lab ingest project/dataset.py 20                       # prints the dataset id
lab run project/experiment.py                          # newest dataset of its source; --dataset <source or id prefix>
lab run project/experiment.py ideas/agent7.py          # pipelines from both files, the evaluations from one
lab run project/experiment.py --dry-run                # every id computed, nothing written: what would fit and why
lab run project/test.py                                # the test evaluation, once, after the pick is written down
```

Modules are `.py` paths or dotted names importable from the current directory. `lab run` prints one line per feature set, fit, prediction set and score as it writes them, a summary of computed and reused work, and exits 1 naming any pipeline that raised; the failure is recorded with its traceback and the other pipelines continue. The dry run names, per stale fit, which repo files moved against the newest earlier fit of that pipeline, so a run's cost is known before it is paid.

## Read

The ledger is one SQLite file, `ml_lab.sqlite` under the root, with views over the event log. `sqlite3 -box` or DuckDB's `ATTACH ... (TYPE sqlite)` both work. Five queries to start from:

```sql
-- how each pipeline did at the first horizon; fold_mean first, since the pooled value differs for a ratio metric
SELECT name, metric, fold_mean, fold_std, folds, value FROM board
WHERE evaluation_name = 'validation' AND window = '1' ORDER BY metric, fold_mean DESC;

-- every pipeline against one reference, fold by fold, with the fold-paired t
SELECT name, mean_delta, delta_std, t, wins, folds FROM head_to_head
WHERE reference_name = 'ridge_3m' AND evaluation_name = 'validation' AND window = '1' AND metric = 'pnl'
ORDER BY mean_delta DESC;

-- one pipeline fold by fold
SELECT f.fold, f.label, f.window, f.metric, f.value FROM score_latest l
JOIN score_fold f ON f.score = l.score WHERE l.name = 'ridge_3m' ORDER BY f.fold, f.window, f.metric;

-- what failed, and in which run
SELECT p.name, x.error, x.run, x.at FROM raw_failure x JOIN raw_pipeline p ON p.id = x.pipeline;

-- what each scored pipeline cost to fit
SELECT l.name, COUNT(*) AS fits, SUM(sf.duration_s) AS seconds FROM score_latest l
JOIN score_fit sf ON sf.score = l.score GROUP BY l.score ORDER BY seconds DESC;
```

`board` and `head_to_head` are the reading views; `raw_<event>` is one event type as columns, for joins; `score_fold`, `score_aggregate`, `score_latest` and `score_fit` are the building blocks. The scorer's series blobs are there too (`score_aggregate.series`), for a row-paired test or a bootstrap in a notebook. The event and view tables, the blob formats, the designs declined with their reasons and what is deferred are in [docs/spec.md](docs/spec.md).

## What refuses, and why

The refusals are the product. The ones you will meet first:

- **"fold windows train on labels revealed after they start"**: a fold's train rows carry labels not known at its window's start. The message names the worst fold and how many more timestamps the embargo must drop; one dry run sizes it.
- **"read past row N"**: a function's output changed when recomputed on a shorter prefix, so it reads later rows or is not deterministic. The first moved row, both values and the size of the move are in the message; rounding is allowed up to 1e-9 of scale.
- **"source changed during the run"**: a module a pipeline's function imports was edited while the run was fitting. Rerun.
- **"the model loaded from its saved bytes predicts differently"**: `save` does not keep what `predict` reads.
- **"is defined in ..., which the fit function does not import"**: the config dataclass lives in a module its function does not import, so an edited default would escape the memo.
- **"is not a module-level function of ..."**: a slot holds a lambda, a closure, a method or a partial. The log names functions by dotted path, so define it at module level and pass its knobs through `config` or `postprocess_config`.
- **"yields other folds than evaluation ... recorded"**: the split's code changed under an unchanged declaration. Rename it or change a field, and the old evaluation and its scores stay.
- **"ledger schema N, this build is M"**: the log is a cache of code plus data; delete the root and rerun.

## Layout

`src/ml_lab/ledger.py` (the event table, its views, the blob store), `experiment.py` (Pipeline, Scorer, Evaluation), `splits.py`, `dates.py`, `runs.py` (the memo, the guards, the probes), `dataset.py` (Dataset, the resident rows; record, load), `formats.py`, `identity.py` (content hashes, code keys, import closures, the environment lock), `panel.py` (a (time, key) grid over the rows), `cli.py` (`lab`). Dependencies: numpy and polars.

Status: used in the `ml-lab-exp` studies (order-book signals, equity cross-sections, Kaggle tabular problems) by people and by agents. The API still moves where those studies push on it, and every push lands as a numbered friction issue and a ruling in the pull request that closes it. Background: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md).
