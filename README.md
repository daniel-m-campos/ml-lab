# ml-lab

A local-first record of experimentation on frozen datasets whose rows are in a fixed order: time series when there is a clock, any tabular data once shuffled into one. For one person or a team of agents working the same data. You declare pipelines and evaluations in plain Python; `lab run` fits, predicts and scores only what the log does not already hold, refuses the common leaks before they are recorded, and writes every result as an event that names the code, data and environment behind it. You read the results with SQL.

It exists because experiment tracking tools record what you tell them, and the things that go wrong (a feature that reads the future, a label known a day later than its row, a k-fold fit that saw its validation labels, a fit rerun under edited code, two "identical" runs with different numbers) are not things anyone tells them. Here identity is computed from content, the guards run on every write, and the reads are tables.

## What you get

- **A memo over everything.** A fit is reused when one exists for the same dataset, declaration, train range, code and environment; so are predictions and scores. A rerun of an unchanged tree writes nothing and says what it reused. Adding one pipeline costs only that pipeline.
- **Identity from content.** A dataset id is its rows; a pipeline id is its declaration; a fit id covers the code keys (syntax trees, so comments and formatting do not count) of every module its steps import, plus the lock of every distribution they reach. Names are labels and never enter an id.
- **Leak guards that refuse, not warn.** Every step sees a prefix of the data ending where it may look. Labels not yet known at a window's start are NaN inside it. A fold that trains on labels revealed after its window starts is refused before anything runs, with the embargo that clears it. Every computed window is recomputed on a shorter prefix and must agree, so a step that reads later rows is refused with the first row that moved.
- **Provenance per number.** A score names its run, predictions and fits; a run records the commit, the dirty diff, the host and the tool's own commit; a fit records its import shas and environment lock.
- **One ledger, many writers.** Agents and people append to the same SQLite file under an actor name. A research repo with one directory per study and one shared ledger is the pattern the `ml-lab-exp` studies use.

## Install

```
git clone git@github.com:daniel-m-campos/ml-lab.git
uv pip install -e ml-lab                 # editable, into the project's venv; Python 3.12, numpy and polars
```

Editable is the point: the environment lock hashes the tool's code, so a `git pull` that changes how a fit is computed refits what it touched and a pull that changes docs refits nothing. A non-editable install carries only the version string and would reuse fits across tool changes. Upgrading is `git pull`; a ledger survives it unless the schema version moves, which `lab` refuses loudly with the instruction to delete and rerun. Every run records the tool's commit and dirty diff (`SELECT editable FROM raw_run`), so an issue can name the exact tool it hit. Feedback goes in as a GitHub issue with the `friction` label, with a minimal repro; fixes land by pull request and bump the patch version.

For development: `uv venv .venv && uv pip install -e ".[dev]"` then `.venv/bin/python -m pytest`. The suite is the ledger story on synthetic data, `tests/test_ledger_story.py`, and every refusal the tool makes is pinned there.

## A project

A project is a git repository exposing three things, in one file or several. The Optiver example under `examples/optiver/` is the template; `examples/optiver/launch.sh 20` runs it end to end on twenty stocks when the Kaggle "Trading at the Close" `train.csv` sits under `data/optiver/optiver-trading-at-the-close/`.

**`dataset.py`: how the raw data becomes a session.** A `Session` is a dict of equal-length numpy columns plus an optional sorted `ts` clock. `record` applies the filters, stores the rows as Parquet and returns the dataset id, which is the data: a loader fix that changes rows is a new dataset, and a re-ingest of identical rows writes nothing.

```python
from ml_lab.dataset import record
from ml_lab.experiment import step
from ml_lab.session import Session

@step
def drop_null_target(session: Session) -> Session:
    keep = ~np.isnan(session.columns["target"])
    return Session({k: v[keep] for k, v in session.columns.items()}, session.ts[keep])

def dataset(ledger, stocks: str = "all") -> str:
    return record(ledger, load(stocks), source="optiver", params={"stocks": stocks},
                  filters=(drop_null_target,), targets=("target",),
                  reveal={"target": datetime.timedelta(seconds=60)})
```

`targets` names the label columns, which the tool masks where a step may not see them. `reveal` says when a label is known: a `timedelta` after its row, or an integer `n` meaning the first row of the n-th later date with rows (`1` is the next trading day's open). Without it a label is known right after its own timestamp. Targets must be numeric or boolean.

**`steps.py`: features, models and scorers.** A `Pipeline` has slots, each a function registered with `@step`, and the slots are scopes: `features` sees the whole dataset without its targets, `fit` sees the rows up to the end of its train range, `predict` and `postprocess` see the rows up to the end of their window. A step belongs in the broadest scope of what it reads.

```python
from ml_lab import formats
from ml_lab.experiment import scorer, step
from ml_lab.session import Range, Session
from ml_lab.splits import Segments

@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    train_window_months: int
    alpha: float

@step
def ridge_fit(session: Session, train: Segments, config: RidgeConfig) -> RidgeModel:
    rng = last_months(session, train, config.train_window_months)
    X, y = features(session, rng), session.column("target", rng)
    ...
    return RidgeModel(weights, bias)

@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    return features(session, rng) @ model.weights + model.bias      # (rows,) or (rows, k)

@step(format=formats.Format.ARROW_ARRAYS)
def ridge_save(model: RidgeModel) -> bytes:
    return formats.arrays_save({"weights": model.weights, "bias": np.array([model.bias])})

@step
def ridge_load(payload: bytes) -> RidgeModel:
    arrays = formats.arrays_load(payload)
    return RidgeModel(arrays["weights"], float(arrays["bias"][0]))

@scorer(metrics=sim_metrics, directions={"pnl": "max", "max_dd": "min"}, columns=("pnl", "flips"))
def taker_sim(pred: np.ndarray, session: Session, rng: Range, config: SimConfig) -> np.ndarray:
    truth = session.column("target", rng)
    position = np.sign(pred) * (np.abs(pred) > config.threshold_bps)
    flips = np.abs(np.diff(position, prepend=0.0))
    return np.column_stack([position * truth - config.cost_bps * flips, flips])
```

`session.column(name, rng)` and `session.matrix(rng, names)` read a row range; a read past the step's prefix raises. A `fit` sees its targets as NaN outside its train segments, so a k-fold fit cannot read its validation block. `save` declares a format from `formats.KNOWN`: `arrow-arrays` and `zip` open without Python, `pickle` is admitted for scikit-learn and marked non-portable, and `bonsai-msgpack` goes through a temp-file round trip. `predict` returns one row per window row, `(rows,)` or `(rows, k)` for a multi-class model, and the stored blob and the scorer carry that width. A scorer returns a series with one row per scoring unit and `metrics(series)` reduces it, per fold and over the concatenated folds; the series is stored, so a later analysis (a paired test, a bootstrap) reads it without the scorer's code.

The optional slots: `features(session) -> {name: array}` adds columns computed once per dataset and shared by every pipeline that declares the step, read by the names in `session.feature_columns`; `postprocess(predictions, session, rng)` is the cheap stateless stage after `predict` (neutralise, clip, rank), whose knobs are bound with `step.configured(at=3.0)` so they enter the prediction's identity and not the fit's. A pipeline with `members` is a blend: its `fit` and `predict` take the members' predictions as a fourth argument.

A config is a frozen dataclass defined in a module its step imports, so an edited default is caught by the memo. Keep each model family, the baselines, the feature step and the scorer in their own modules: the unit of invalidation is a step's module and everything it imports.

**`experiment.py`: the pipelines and the evaluations.**

```python
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.splits import CalendarWalkForward

ridge = Pipeline(name="ridge_3m", fit=steps.ridge_fit, predict=steps.ridge_predict,
                 save=steps.ridge_save, load=steps.ridge_load,
                 config=steps.RidgeConfig(train_window_months=3, alpha=1.0))
pipelines = [ridge.with_config(train_window_months=m).named(f"ridge_{m}m") for m in (1, 3, 6)]

def evaluations(dataset: str) -> list[Evaluation]:
    return [Evaluation(name="validation", dataset=dataset,
                       split=CalendarWalkForward(first_cutoff="2021-05-04", horizons=(1, 2, 3),
                                                 embargo_timestamps=1, min_folds=3),
                       scorer=steps.taker_sim, config=steps.SimConfig(cost_bps=0.5))]
```

Variants are `dataclasses.replace` (or `with_config` and `named`): a pipeline that differs only in `postprocess` shares every fit and raw prediction with its parent. A split is any frozen dataclass with `folds(session)`; the shipped ones are `CalendarWalkForward` (cutoffs every N months or trading days on the clock, named horizons, an embargo in whole timestamps), `WalkForward` by rows, and `Holdout(train_fraction)` with `BlockedKFold(k, train_fraction)` over the rows it trains on, so a validation and a test evaluation share one cut. Folds are contiguous ranges in the stored row order, so the order chosen at ingest is the split's design: a shuffle in a filter step makes `BlockedKFold` a random k-fold, an interleave by class makes it stratified, and a sort by group with the cut snapped to the group's boundary makes it grouped. Every step sees the rows before where it may look, with or without a clock; a clock adds reveal lags, calendar cutoffs and embargoes in whole timestamps. Keep the test evaluation in its own module (`test.py`), so running the validation module never scores it. Every pipeline is scored under every evaluation in the run.

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
- **"read past row N"**: a step's output changed when recomputed on a shorter prefix, so it reads later rows or is not deterministic. The first moved row, both values and the size of the move are in the message; rounding is allowed up to 1e-9 of scale.
- **"source changed during the run"**: a module a step imports was edited while the run was fitting. Rerun.
- **"the model loaded from its saved bytes predicts differently"**: `save` does not keep what `predict` reads.
- **"is defined in ..., which the fit step does not import"**: the config dataclass lives in a module the step does not import, so an edited default would escape the memo.
- **"yields other folds than evaluation ... recorded"**: the split's code changed under an unchanged declaration. Rename it or change a field, and the old evaluation and its scores stay.
- **"ledger schema N, this build is M"**: the log is a cache of code plus data; delete the root and rerun.

## Layout

`src/ml_lab/ledger.py` (the event table, its views, the blob store), `experiment.py` (Pipeline, Evaluation, step, scorer), `splits.py`, `dates.py`, `runs.py` (the memo, the guards, the probes), `dataset.py` (record, load), `formats.py`, `identity.py` (content hashes, code keys, import closures, the environment lock), `session.py` (the resident rows), `panel.py` (a (time, key) grid over the rows), `cli.py` (`lab`). Dependencies: numpy and polars.

Status: used in the `ml-lab-exp` studies (order-book signals, equity cross-sections, Kaggle tabular problems) by people and by agents. The API still moves where those studies push on it, and every push lands as a numbered friction issue and a ruling in the pull request that closes it. Background: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md).
