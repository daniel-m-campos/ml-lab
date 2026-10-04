# ml-lab: design spec (draft 8, 2026-10-03)

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. People and agents write through two commands and read with SQL. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

Datasets, pipelines, evaluations, fits, predictions and scores are content-addressed objects, so the same work is never done twice and every number names the code, data and environment that produced it. Decisions over scores (a baseline, promote and reject, a board) are designed but deferred to Later: the first job is to analyze the evaluations by hand and let that refine the schema.

## Scope

In: frozen datasets, pipeline and evaluation declarations, runs that memoize fits and predictions, per-fold and aggregate scores, SQL views, a two-verb command line, analytics over the log. Out, for now: decisions and a baseline (Later), automation that moves one, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## Three systems and a boundary

| concern | owner | what the log records |
|---|---|---|
| code | git | per run: commit, dirty flag, the dirty diff as a blob; per fit: the git blob sha of every repo module imported while fitting, and the environment lock |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, diffs, each under its sha, in a format that opens without this Python environment |
| facts | the event log, `ml_lab.sqlite` | the seven event types below |

The log never stores what a function is, only a dotted path and the blob shas git computed for the files. Reproducing a number is the join: the log says which shas and which commit, git has the content, the blob store has the bytes.

## Declarations

An evaluation is a dataset, a split, a scorer and its config. The split is a frozen dataclass with `folds(session)`: `WalkForward` (cutoffs every N rows, a window after each, optional named horizons), `BlockedKFold` and `Holdout` work on any session by row count; `CalendarWalkForward` (cutoffs every `every` months or trading days on the session clock, horizons in `window` units, an embargo in whole timestamps, an optional `end` date) needs `ts`; too few folds is refused with the date arithmetic that ran out. A project may declare its own. A fold trains on contiguous segments and scores named windows. A pipeline is `fit(session, train_segments, config) -> model`, `predict(model, session, range) -> predictions`, and `save(model) -> bytes` with `load(bytes) -> model` naming a format that opens without Python. A scorer is `score(predictions, session, range, config) -> series`, a 1-D or 2-D array with one row per scoring unit (a prediction, or a day for a cross-sectional metric; `metrics` defines the unit and must make sense over the folds' rows concatenated), with `metrics(series) -> {name: value}` and a direction per metric declared once on the scorer; the same `metrics` runs per fold and over the concatenated folds.

`session` is the frozen dataset's columns, resident in memory, read by row range. Every step sees a prefix of it: `fit` up to the end of its last train segment, `predict`, `postprocess` and the scorer up to the end of their window, with row positions unchanged, so a feature cannot read past the cutoff and a read that tries raises. Inside the window, `predict` and `postprocess` see the dataset's target columns as NaN from the window start, and once per pipeline, on its first computed window, the run freezes every row after the window's midpoint and requires the predictions before it to stand; a step that reads later rows, or one that is not deterministic, is refused with the row named. The scorer keeps the real targets. An optional `features(session) -> {name: array}` step runs before `fit` over the whole session without its target columns and adds its arrays as columns, which the steps then read through the same prefix views; on first computation the run freezes every row from the schedule's first cutoff and requires the columns before it to stand, so a feature that reads past its row is refused with the column named. Anything that learns from data lives inside `fit` over `train_range`. Cross-sectional data is still rows: `ml_lab.panel.Panel(session, key)` grids a column by (time, key) once per session and maps a grid back to rows. Declarations are frozen dataclasses referencing functions by dotted path, hashed by canonical serialization: a value change changes the id, a name is a label outside the hash, and a field at its default is left out so adding a defaulted field keeps every id. A dataclass config must be defined in a module its consumer imports (the fit step for a pipeline's, the scorer for an evaluation's), so an edited default is under the memo; one defined elsewhere in the repo is refused. No YAML, no closures. Variants are `dataclasses.replace`: a pipeline that differs only in `postprocess` shares every fit and raw prediction. Identity is declaration only; code changes are caught by the memo rule, not by hashing files into identities.

## Schema

One table. Rows are only inserted.

```sql
CREATE TABLE event (
  seq     INTEGER PRIMARY KEY,   -- global order
  id      TEXT UNIQUE,           -- content hash for objects, ULID for runs and failures
  type    TEXT,                  -- one of eight
  stream  TEXT,                  -- the dataset or evaluation it belongs to
  key     TEXT,                  -- what it is about
  at      REAL,
  actor   TEXT,                  -- "daniel" or "agent:<name>"
  host    TEXT,
  payload TEXT                   -- JSON, one shape per type, versioned
);
```

| type | stream | key | payload |
|---|---|---|---|
| dataset_recorded | dataset | dataset id = hash(column names, dtypes, bytes) | source, window, recipe (params, filter paths, targets) as provenance, rows, blob sha |
| pipeline_declared | pipeline | event id = hash(pipeline id, name); key = pipeline id = hash(declaration) | name, fit, predict, save, load and postprocess paths with bound kwargs, config, member declarations for a blend; the views show the newest name per id |
| evaluation_declared | evaluation | event id = hash(evaluation id, name); key = evaluation id = hash(declaration) | name, dataset id, split, scorer path, config, metric directions, the expanded folds (label, train segments, named windows); a split that yields other folds under an unchanged declaration is refused |
| run_started | evaluation | run id (ULID) | commit, dirty (tracked changes or untracked files), diff sha covering both as `git diff HEAD` plus a new-file diff per untracked file, resolution file sha (`uv.lock` or `requirements*.txt` if present), host facts |
| features_computed | dataset | feature id = hash(dataset, features step, its import shas, its env lock sha) | pipeline id, run id, column names, the probe row, duration, the columns as a Parquet blob |
| fit_computed | dataset | fit id = hash(dataset, fit + save + load steps and config, train segments, their import shas, env lock sha, the feature id when the pipeline has a features step) | run id, label, model sha, format and whether it is portable, import shas, env lock sha, duration; a blend's fit also hashes and records its members' fit ids. A fit, prediction or postprocess whose module files changed on disk between the run's start and its write is refused, so a sha is never recorded for code that did not run |
| predictions_computed | dataset | raw: hash(fit id, range, predict step and its import shas); postprocessed: hash(raw id, postprocess step with kwargs and its import shas) | blob sha and format, fold, window; a postprocessed one names its raw id and step; a blend's names its members' prediction ids, and, only when its `fit` takes a fourth positional parameter, its members' in-sample predictions over the train segments are predictions too, window `train:k` |
| score_recorded | evaluation | score id = hash(evaluation, pipeline, prediction ids, the scorer's import shas) | per-fold metrics, aggregate per window, and per window the scorer's series as an `arrow-arrays` blob with the fold row counts, its columns named by `scorer(columns=...)` or `0`, `1`, ..., so any uncertainty method can be run later |
| pipeline_failed | evaluation | pipeline id | run id, error, traceback sha |

Fits and predictions live on the dataset stream because a scoring change reuses them across evaluations. A score is "this pipeline, under this evaluation, from exactly these predictions"; a rerun that reproduces the same predictions writes no score, and a code change that changes them writes a new one.

### Views

Shipped as SQL in the same file so `sqlite3` shows them as tables. The one-event-type views carry `seq`, so `WHERE seq <= N` reads the log as it stood; `latest_score` ranks over the whole log.

| view | definition |
|---|---|
| dataset, pipeline, evaluation, run, feature, fit, prediction, score | one event type each, payload fields as columns; pipeline and evaluation show the newest name per id |
| fold_score, aggregate_score | `score_recorded` unpacked one row per (fold, window, metric) and per (window, metric); the aggregate row carries the fold count, fold mean and fold standard deviation of the metric and the series blob sha |
| latest_score | per (evaluation, pipeline), the newest score, with the pipeline and evaluation names and the dataset's id and source joined on |
| paired_score | per (evaluation, window, metric) and ordered pair of latest pipelines: the fold count, the mean and sample std of the per-fold difference, and the folds the first wins; `fold_std` on `aggregate_score` is one pipeline's spread across windows, shared by every pipeline, so the noise of a comparison is `delta_std` here |
| score_fit | one row per (score, fit): the fits a score stands on, including a blend's members, with the fit's pipeline, label and seconds |
| failure | `pipeline_failed` with run and error |

The payload is JSON so a shape change is a new payload version and an edited view, not a migration. Materialize a view only when a read is measured past a second.

## Rules

Memo: a fit is reused when one exists for (dataset, pipeline, train range) whose recorded import shas and environment lock all match the current tree. Dirty files hash like any other; the run records the diff. Predictions are reused per (fit, range). Features are reused per (dataset, step, its code, its environment), so one feature set serves every pipeline and fold that declares it. The unit of invalidation is a step's module and everything it imports: keep each model family, the baselines, the feature step and the scorer in their own modules. `lab run --dry-run` computes every id, looks each one up, writes nothing, and names per stale fit the repo files that moved against the newest earlier fit of the same pipeline and label.

The environment lock is a text blob, one `name==version` line per installed distribution the pipeline's import closure reaches, with their requirements, plus the Python version. Installing an unrelated package changes nothing; bumping a dependency refits the pipelines that import it, and two lock blobs diff as text. An editable install's version carries a hash of its source files. Platform is recorded on the run, not in the lock, so a GPU box reuses a laptop's fits.

The import closure is every module under the repo root reachable from the step modules' top-level imports, with `.venv` excluded. `experiment.py` is in it only if a step imports it, so editing a config there changes the pipeline id, not the fit shas. A repo module or a distribution imported lazily inside a step is invisible to the closure; after the first fit the run sees it loaded, refuses, and names it, because the memo would otherwise have missed its edits or its version bumps.

Comparability: scores compare only within one evaluation. A scorer or schedule change is a new evaluation.

Faults: every blob is written before the event that names it, and each fit is its own event, so a killed process loses at most the fit in flight and a rerun resumes from the last recorded one with the same ids. A pipeline that raises is recorded as `pipeline_failed` with its traceback, the other pipelines continue, and `lab run` exits 1 naming it. Failures are not memoized: a rerun retries. The `failure` view holds the error until a newer score exists.

Splits: a fit's train segments never overlap its fold's windows, and the embargo the split declares separates them. The folds are in the evaluation payload, written once and checked against the live split on every run; `session.ts` is optional and only `CalendarWalkForward` needs it.

Day one, because rows written wrong cannot be repaired: ids that merge across hosts (content hashes and ULIDs, no serial counters); the blob store writes to a temp file, fsyncs and renames; every fit carries its import shas and env lock. Two logs from two hosts merge by `INSERT OR IGNORE` on id. The schema version is `PRAGMA user_version`; a log at another version is refused with the instruction to delete and rerun, since the log is a cache of the code plus the data.

## Blobs

Every sha reference in a payload carries a `format` from `formats.KNOWN`, which also says whether the format opens without this Python environment. Every shipped format does except `pickle`, admitted because scikit-learn has nothing else; a fit whose model is a pickle carries `portable = false`, so `SELECT ... FROM fit WHERE NOT portable` names the models hostage to the environment. A save step declaring a format outside the set is refused at run.

| blob | format name | bytes |
|---|---|---|
| dataset rows | `parquet` | Parquet, `ts` as a nanosecond timestamp column plus one float64 column per field |
| predictions | `parquet` | Parquet, one column `prediction`; the row range is in the event |
| model | what the pipeline's `save` step declares | `save(model) -> bytes` and `load(bytes) -> model` as registered steps; `formats.py` ships `arrow-arrays` (a dict of arrays as an Arrow IPC file) and the temp-file round trip the Optiver example uses for `bonsai-msgpack`; a pipeline without them is refused at run |
| dirty diff | `text/x-diff` | the `git diff HEAD` text |
| declarations | not a blob: the JSON payload, re-imported by dotted path |

A blob's name is the sha256 of its bytes as stored, so identical content is written once and `fsck` can rehash and compare. The database cannot enforce a pointer into the filesystem; a missing blob is found by read or by `fsck`.

## Analytics

SQLite is the record; DuckDB is the analyst. `ATTACH 'ml_lab.sqlite' (TYPE sqlite)` queries the same views with columnar speed and hands frames to polars or pandas. Per-fold and aggregate metrics come from the views; per-row work (paired tests on pnl) loads prediction blobs by sha. Nothing in the log is redesigned for analytics.

## From inception to experimentation

1. A project is a git repo exposing three module attributes, in one file or several: `dataset(ledger, *args)`, `pipelines` and `evaluations` (a list, or a function of the dataset id returning one; every pipeline is scored under each, so a scoring sweep is one list of `dataclasses.replace(base, config=..., name=...)`). `lab` reads them by name, so a file holding only new pipelines runs beside the project's declarations.
2. `lab ingest` appends `dataset_recorded` and stores the rows under their sha. The id is the data, so a loader fix that changes rows is a new dataset and a re-ingest of identical rows writes nothing.
3. The first `lab run` appends `evaluation_declared` with the schedule expanded once.
4. A `lab run` with work appends `run_started`, then only the fits and predictions the memo rule does not cover, then one `score_recorded` per pipeline whose predictions are new. A rerun of an unchanged tree writes nothing and prints what it reused; adding one pipeline costs only that pipeline's fits, predictions and score.
5. Read with SQL: the latest score per pipeline, its aggregate and per-fold scores, the failures. Add a `Pipeline`, `lab run`, query again.
6. Code evolves: changed import shas refit, new scores appear, old scores stand with their commit and shas.
7. Reproduce: the score names its run and predictions, the run its commit and resolution file, the fit its shas, lock and blob. Check out, sync, load, rerun, compare shas.

## Command line

```
lab ingest project/dataset.py <args>              # prints the dataset id
lab run project/experiment.py                   # newest dataset of the one source; --dataset <source or id prefix>
lab run project/experiment.py ideas/agent7.py   # pipelines from both, the evaluations from one
lab run project/experiment.py --dry-run         # what a run would compute and why; writes nothing
sqlite3 -box .ml-lab/ml_lab.sqlite "..."
```

Two verbs. A module is a dotted name importable from the current directory or a `.py` path, resolved by its package so its own imports work. The ledger root is `ML_LAB_ROOT` or `--root`; only `lab ingest` creates a ledger, and `lab run` against a root without one refuses with the resolved path. Every write carries the actor.

Reads are SQL over the views. Four to start from:

```sql
-- latest aggregate scores per pipeline at the first age
SELECT l.name, s.metric, s.value FROM latest_score l
JOIN aggregate_score s ON s.score = l.score
WHERE s.window = '1' ORDER BY s.metric, s.value DESC;

-- every pipeline against one reference, fold by fold
SELECT name, mean_delta, delta_std, wins, folds FROM paired_score
WHERE reference_name = 'zero' AND window = '1' AND metric = 'pnl' ORDER BY mean_delta DESC;

-- one pipeline fold by fold
SELECT f.fold, f.label, f.window, f.metric, f.value FROM latest_score l
JOIN fold_score f ON f.score = l.score
WHERE l.name = 'ridge_3m' ORDER BY f.fold, f.window, f.metric;

-- what failed, and in which run
SELECT p.name, x.error, x.run, x.at FROM failure x JOIN pipeline p ON p.id = x.pipeline;

-- what each scored pipeline cost to fit
SELECT l.name, COUNT(*) AS fits, SUM(sf.duration_s) AS seconds FROM latest_score l
JOIN score_fit sf ON sf.score = l.score GROUP BY l.score ORDER BY seconds DESC;
```

## Later

| item | reopener |
|---|---|
| decisions: a baseline per evaluation, promote and reject on a score, a board with Pareto verdict and relative deltas, history | when reading the SQL by hand stops being how the evaluations are judged. Design kept: a score is content-addressed by its prediction ids, so a decision pins to content and new content needs a new decision; the baseline is the last promote; nothing is deleted, a retraction is a new decision. Stories S3 and S6 describe it |
| ordering rule for incomparable sets | when reading deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation |
| stacked blends: members fit on an inner split so the combiner learns on out-of-fold predictions | the first learned combiner wanted; today a blend's `fit` with four positional parameters receives the members' in-sample train-range predictions, a three-parameter `fit` receives none and sets fixed weights |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation |
| deploy, bundle, refresh seating | when a pipeline goes to production from the log |
| materialized projections, typed payloads | when a view read passes a second, or the payload shapes stop moving |
| remote executors, agent journal, lineage export, a blob remote | when a run leaves the laptop |
