# ml-lab: design spec (draft 8, 2026-10-03)

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. People and agents write through two commands and read with SQL. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

Datasets, pipelines, evaluations, fits, predictions and scores are content-addressed objects, so the same work is never done twice and every number names the code, data and environment that produced it. Decisions over scores (a baseline, promote and reject, a board) are designed but deferred to Later: the first job is to analyze the evaluations by hand and let that refine the schema.

## Scope

In: frozen datasets, pipeline and evaluation declarations, runs that memoize fits and predictions, per-fold and aggregate scores, SQL views, a two-verb command line, analytics over the log. Out, for now: decisions and a baseline (Later), automation that moves one, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## Three systems and a boundary

| concern | owner | what the log records |
|---|---|---|
| code | git | per run: commit, dirty flag, the dirty diff as a blob; per fit: the git blob sha of every repo module the pipeline's steps import, and the environment lock |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, diffs, each under its sha, in a format that opens without this Python environment |
| facts | the event log, `ml_lab.sqlite` | the nine event types below |

The log never stores what a function is, only a dotted path and the blob shas git computed for the files. Reproducing a number is the join: the log says which shas and which commit, git has the content, the blob store has the bytes.

## Declarations

An evaluation is a dataset, a split, a scorer and its config. A split is a frozen dataclass with `folds(session)`; a project may declare its own. A fold trains on contiguous segments and scores named windows.

Row splits work on any session: `WalkForward` (cutoffs every N rows, a window after each, optional named horizons), `Holdout(train_fraction)`, and `BlockedKFold` over the rows that `Holdout` trains on, so a validation and a test evaluation share one cut. On a session with a clock every row cut lands on the first row of its timestamp, so a cross-section is never split between train and a window; an embargo stays a row count. A window named `test` is scored, not held out; the evaluation's name says validation or test. `CalendarWalkForward` needs `ts`: cutoffs every `every` months or trading days, horizons in `window` units, an embargo in whole timestamps, an optional `end`. A window ends on a unit boundary at or before `end`, and the day after the data counts as `end`, so the last trading day is scored. Too few folds is refused with the date arithmetic that ran out; an empty window, a fold with no train rows, or a field below its floor is refused by name.

A pipeline is `fit(session, train_segments, config) -> model`, `predict(model, session, range) -> predictions`, and `save(model) -> bytes` with `load(bytes) -> model` naming a format that opens without Python, plus the optional `features` and `postprocess` steps. The slots are scopes, not a chain: `features` is dataset scope (once per dataset and step), `fit` is fold scope (once per train range), `predict` and `postprocess` are window scope (once per fit and range). A step belongs in the broadest scope of what it reads, which is what makes each memo key and leakage guard hold; composition within a scope is the object `fit` returns.

A scorer is `score(predictions, session, range, config) -> series`, a 1-D or 2-D array with one row per scoring unit (a prediction, or a day for a cross-sectional metric, as `metrics` defines it), with `metrics(series) -> {name: value}` and a direction per metric declared once on the scorer. The directions travel with each score: an edited direction is scorer code, so it writes a new score, and `head_to_head` counts wins by the directions its scores carry (a score recorded without them reads the evaluation's first declaration). The same `metrics` runs per fold and over the concatenated folds, and the pooled value equals the row-weighted fold mean only for a metric that adds over rows. `metrics(series)` reruns on the stored blob alone, so the series carries what `metrics` needs (a constant column, such as a holding period, is the documented way) and per-row config such as costs goes in the scorer body.

`session` is the frozen dataset's columns, resident, read by row range; NaT or unsorted timestamps are refused. Every step sees a prefix with row positions unchanged: `fit` to the end of its last train segment, `predict`, `postprocess` and the scorer to the end of their window. A read past it raises, as does a segment below row 0 or ending before it starts. `fit` sees every target NaN outside its train segments, features untouched, so a k-fold fit cannot read its window's labels. The embargo still matters: a feature built from revealed labels, on a train row after a k-fold window, can carry that window's labels, and only an embargo drops the row. Inside a window, `predict` and `postprocess` (an `in_sample` member's train-segment predict included) see every label not known at the window start as NaN. A time lag reveals at the row's timestamp plus the lag; an integer lag `n` at the first row of the n-th later date with rows (`1` is the next trading day's open); no lag, only after the row's own timestamp. Targets reach fit and predict as float64. The scorer keeps the real targets and sees dataset columns only.

Before writing a computed window, the run reruns the step on the rows before the first row of the window's middle timestamp, array inputs cut there, and requires the earlier predictions to stand; a window of one timestamp has no cut and is not probed. Rows sharing a timestamp are one instant, so a same-time cross-section passes. Standing means within `1e-9` of the window's largest value (1024 ulps for float32), NaNs equal: honest BLAS and layout differences stay under 3e-12 of scale, the smallest measured leak is 1e-7, and a per-element ulp rule is unsound near zero. A step that reads later rows or is not deterministic is refused with the first moved row, both values, the moved count and the largest move as a fraction of scale, and nothing is recorded. `predict` and `postprocess` return one value per window row; another shape is refused, naming the stage and both shapes, before the probe. A fit is recorded only after its reloaded model predicts its first window as the fitted one did, under the same rule.

An optional `features(session) -> {name: array}` step, or a tuple of them, runs before `fit` over the whole session without its target columns (a target with a declared reveal lag is included) and stores its arrays as columns, which the steps read through `session.column` and `session.matrix` by the names in `session.feature_columns`, in step order. A step that redefines a dataset column or returns an array of another shape than `(rows,)` is refused, as are two steps naming one column (one step listed twice included). Feature columns are not in `session.columns` but read on demand from their stored file, so a column no step names costs no memory, and a learned selection lives in `fit` and pays only for the columns it keeps. On first computation the run reruns each step on five prefixes cut at timestamp boundaries at sixths of the dataset, with revealed labels not yet known masked, and requires every column to match, so a feature that reads past its row is refused with the column, row and both values named, and one that reads a label before its lag is refused saying so; a step whose column names change with the prefix is refused with both sets and the probe row. A blend without its own `features` sees its members' feature columns when every member declares the same steps. Anything that learns from data lives inside `fit` over `train_range`. Cross-sectional data is still rows: `ml_lab.panel.Panel(session, key)` grids a column by (time, key) once per session and maps a grid back to rows, and refuses two rows sharing one (time, key).

Declarations are frozen dataclasses referencing functions by dotted path, hashed by canonical serialization: a value change changes the id, a name is a label outside the hash, and a field at its default is left out so adding a defaulted field keeps every id. `features=(f,)` is `features=f` and `features=()` is no features, so each pair is one id. A dataclass config must be defined in a module its consumer imports (the fit step for a pipeline's, the scorer for an evaluation's), so an edited default is under the memo; one defined elsewhere in the repo is refused. No YAML, no closures. Variants are `dataclasses.replace`: a pipeline that differs only in `postprocess` shares every fit and raw prediction. Identity is declaration only; code changes are caught by the memo rule, not by hashing files into identities.

## Schema

One table. Rows are only inserted.

```sql
CREATE TABLE event (
  seq     INTEGER PRIMARY KEY,   -- global order
  id      TEXT UNIQUE,           -- content hash for objects, ULID for runs and failures
  type    TEXT,                  -- one of nine
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
| dataset_recorded | dataset | dataset id = hash(`ts` bytes, column names, dtypes, values (numeric and datetime by bytes, strings by value, other objects refused), targets with reveal lags) | source, window, recipe (params, filter paths, targets with reveal lags in seconds or as a count of dates) as provenance, rows, blob sha. A session with no rows, a column named `ts` (the clock's name), a missing target, a target that is not numeric or boolean (targets are masked with NaN), a reveal lag naming a non-target, a reveal lag without a clock, or a lag that is negative or neither a timedelta nor an int is refused. A source is a label: `--dataset <source>` resolves to its newest dataset only while one actor wrote under it, and two actors under one source are refused with both ids |
| pipeline_declared | pipeline | event id = hash(pipeline id, name); key = pipeline id = hash(declaration) | name, fit, predict, save, load and postprocess paths with bound kwargs, config, member declarations for a blend; the views show the newest name per id, and one run declaring one id under two names is refused |
| evaluation_declared | evaluation | event id = hash(evaluation id, name); key = evaluation id = hash(declaration) | name, dataset id, split, scorer path, config, metric directions as first declared (comparisons read each score's), the expanded folds (label, train segments, named windows); a split that yields other folds under an unchanged declaration is refused |
| run_started | evaluation | run id (ULID) | commit, dirty (tracked changes or untracked files), diff sha (one `git diff HEAD` over a temporary index holding untracked files; one over 1 MiB is named, not diffed), resolution file sha (`uv.lock` or `requirements*.txt`), host facts, the run's pipelines (id, name), and `editable`: per editable install the steps reach, its git root's commit, dirty flag and diff |
| features_computed | dataset | feature id = hash(dataset, one features step, its code keys, its env lock sha) | pipeline id, run id, column names, the columns id (ordered names, dtypes and values, strings by value), import shas and code keys, the probe rows, duration, the columns as a Parquet blob; a tuple of steps is one event per step |
| fit_computed | dataset | fit id = hash(dataset, fit + save + load steps and config, train segments, their code keys, env lock sha, the feature columns id when the pipeline has features: the hash of its steps' column ids in order) | run id, label, model sha, format and whether it is portable, import shas, env lock sha, duration; a blend's fit also hashes and records its members' fit ids. A feature set, fit, prediction or postprocess whose module files changed on disk between the run's start and its write is refused, so a sha is never recorded for code that did not run |
| predictions_computed | dataset | raw: hash(fit id, range, predict step and its code keys); postprocessed: hash(raw id, postprocess step with kwargs and its code keys) | blob sha and format, fold, window; a postprocessed one names its raw id and step; a blend's names its members' prediction ids, and, only when the blend declares `in_sample=True`, its members' in-sample predictions over the train segments are predictions too, window `train:k` |
| score_recorded | evaluation | score id = hash(evaluation, pipeline, prediction ids, the scorer's code keys) | per-fold metrics, aggregate per window (a non-finite metric is stored as null), the scorer's metric directions, and per window the scorer's series as an `arrow-arrays` blob with the fold row counts, its columns named by `scorer(columns=...)` or `0`, `1`, ..., so any uncertainty method can be run later; a `columns` count the series disagrees with fails the pipeline as `pipeline_failed` |
| pipeline_failed | evaluation | pipeline id | run id, error, traceback sha |

Fits and predictions live on the dataset stream because a scoring change reuses them across evaluations. A score is "this pipeline, under this evaluation, from exactly these predictions"; a rerun that reproduces the same predictions writes no score, and a code change that changes them writes a new one.

### Views

Shipped as SQL in the same file so `sqlite3` shows them as tables. The names say the layer: `board` and `head_to_head` are for reading and carry no prefix; `raw_` is one event type as columns, for joins; `score_` is a building block over scores. `.tables raw%` lists a layer. The `raw_` views carry `seq`, so `WHERE seq <= N` reads the log as it stood, except `raw_pipeline` and `raw_evaluation` (latest name), `raw_failure` (dropped once a newer score exists) and `score_latest`, which read the whole log.

| view | definition |
|---|---|
| board | how each pipeline did: per (evaluation, pipeline, window, metric) the latest score's `fold_mean`, `fold_std`, `folds` and `value`, in that order, with the dataset source and the evaluation and pipeline names first and the ids last; `score_latest` joined to `score_aggregate` |
| head_to_head | is A better than B: per (evaluation, window, metric) and ordered pair of latest pipelines, with the evaluation name and dataset source: `folds`, those where both metrics are non-null; `mean_delta`, the mean per-fold difference (the difference of the pair's `fold_mean`); `delta_std`, its sample std, NULL at one fold; `t`, fold-paired, folds - 1 degrees of freedom, independent folds assumed, NULL at one fold or zero spread; `wins`, the folds the first wins by the score's direction; `pooled_delta`, the difference of pooled `value`. A comparison's noise is `delta_std`, not `board`'s `fold_std`; at one fold, pair the series (Analytics) |
| raw_dataset, raw_pipeline, raw_evaluation, raw_run, raw_feature, raw_fit, raw_prediction, raw_score, raw_failure | one event type each, payload fields as columns; `raw_evaluation.config` is canonical, so a cost sweep's rows differ without names; `raw_run` carries `editable` and `pipelines`; `raw_failure` carries the run and error until a newer score for its evaluation and pipeline |
| score_fold, score_aggregate | `score_recorded` unpacked per (fold, window, metric) and per (window, metric). `value` is `metrics` over the concatenated folds, `fold_mean` the mean of fold values; a null fold (a non-finite metric) is left out of `folds`, `fold_mean` and `fold_std`. For a metric that does not add over rows (AUC, a rank metric, a correlation, a Sharpe) they differ and can disagree in sign; a constant predictor whose constant moves across folds scores a pooled AUC off 0.5. So `board` lists `fold_mean` first. The aggregate row also carries `folds`, `fold_std`, the series blob sha and its `fold_rows` |
| score_latest | per (evaluation, pipeline), the newest score, with the pipeline and evaluation names and the dataset's id and source joined on. A label is not an id: a declaration change under one name lists the name once per id, and the log cannot say which declaration the code holds (a rerun that reuses everything writes nothing), so a board filters on the id `lab run` prints |
| score_fit | one row per (score, fit): the fits a score stands on, including a blend's members, with the fit's pipeline, label and seconds |

The payload is JSON so a shape change is a new payload version and an edited view, not a migration. Materialize a view only when a read is measured past a second.

## Rules

Memo: a fit is reused when one exists for (dataset, fit declaration, train range) whose recorded code keys and environment lock all match the current tree. A module's code key is a hash of its syntax tree with comments, layout, line numbers and docstrings left out, so a formatter pass, a comment or a docstring edit keeps every id; its git blob sha is recorded beside it as what ran. The one step outside the memo is one whose output reads its own source text, `__doc__` or line numbers, as under `python -OO`. A fit with features hashes the feature columns it consumed (ordered names, dtypes, values), not the feature code, so a feature refactor that keeps the columns keeps every fit. Dirty files hash like any other; the run records the diff. Predictions are reused per (fit, range). An `in_sample=True` blend's fit also hashes the ids of its members' in-sample predictions, so a member whose `predict` or `postprocess` changed refits the blend. Within one run a member is one stage per pipeline id, so a member listed twice or shared by two blends fits once. Features are reused per (dataset, step, its code, its environment), so one feature set serves every pipeline and fold that declares it, and a pipeline declaring several steps shares each with the pipelines that declare it alone. The unit of invalidation is a step's module and everything it imports: keep each model family, the baselines, the feature step and the scorer in their own modules.

Dry run: `lab run --dry-run` computes every id, looks each up and writes nothing. It refuses a moved split as a real run does; it computes but does not store the features it needs, without the five-prefix probe, so every id is real but a feature a real run refuses can pass; it counts shared work once and a ledger hit as one reuse; per stale fit it names the repo files that moved against the newest earlier fit of that pipeline and label; over several evaluations it prints a total.

A declaration error (an unregistered step, a slot holding something that is not a step, a config class its step does not import, a blend whose `fit` arity disagrees with `in_sample`, `in_sample` without members), at any depth of blends inside blends, is refused before anything is read or written. A pipeline that fails under one evaluation is named where it fails, skipped for the rest of that `lab run` and named again at the end; failures are not memoized, so a rerun retries it.

The environment lock is a text blob, one `name==version` line per installed distribution the pipeline's import closure reaches, with their requirements, plus the Python version. Installing an unrelated package changes nothing; bumping a dependency refits the pipelines that import it, and two lock blobs diff as text. An editable install's modules join the closure as the repo's do. Its lock version carries a hash of the code keys of the modules the closure reaches (all its modules when none, as for one reached only as a requirement), so an edit to the tool's command line keeps every fit. Its sources are read once per process, so one `lab run` fits under one lock even if the install is edited meanwhile; `run_started.editable` records its commit and diff. Platform is recorded on the run, not in the lock, so a GPU box reuses a laptop's fits.

The import closure is every module under the repo root reachable from the step modules' top-level names and the loaded modules their import statements name (a module imported for one constant included), through namespace packages under the root, with `.venv` excluded. `experiment.py` is in it only if a step imports it, so editing a config there changes the pipeline id, not the fit shas. A repo module or a distribution imported lazily inside a step is invisible to the closure; the first computation of each stage (features, fit, predict, postprocess) sees it loaded, refuses before writing, and names the stage and the module, because the memo would otherwise have missed its edits or its version bumps.

Comparability: scores compare only within one evaluation, and `head_to_head` pairs only there. A schedule change, a scorer path or config change, or a refit cadence is a new evaluation: an online variant that refits every N days is a split with growing train ranges, not a pipeline property, so it lives beside the house evaluation and pairs with it at the series level (Analytics) when the windows concatenate to the same rows. A scorer code edit is a new score under the same evaluation, since the score id hashes the scorer's code keys and the evaluation id its path. Two studies share an evaluation by importing one scorer module, kept in its own module; a copy at another path is another evaluation. The scorer's `metrics` and `directions` are code, under the memo, not in the evaluation id; each score carries the directions it was scored under. Several folds over one train range share one fit, since the fit id hashes the train range and not the label, so slicing a window into folds costs no fits; slices of one day are serially dependent, and their spread is a floor on the noise, not the noise.

Faults: every blob is written before the event that names it, and each fit is its own event, so a killed process loses at most the fit in flight and a rerun resumes from the last recorded one with the same ids. A pipeline that raises is recorded as `pipeline_failed` with its traceback, the other pipelines continue, and `lab run` exits 1 naming it. `raw_failure` holds the error until a newer score exists.

Splits: the embargo the split declares separates a fit's train segments from its fold's windows. A fold that trains on a row whose label is not known at one of its windows' start, by the dataset's reveal lag, is refused before the evaluation is declared, dry runs included. The message names the worst fold, the lag, and how many more timestamps and rows the embargo must drop to clear every fold, so one dry run sizes the embargo. A fold whose train segments overlap one of its windows is refused the same way, naming the fold, the segment and the window. The folds are in the evaluation payload, written once and checked against the live split on every run; `session.ts` is optional and only `CalendarWalkForward` needs it.

Day one, because rows written wrong cannot be repaired: ids that merge across hosts (content hashes and ULIDs, no serial counters); the blob store writes to a temp file, fsyncs and renames; every fit carries its import shas and env lock. Two logs from two hosts merge by `INSERT OR IGNORE` on id. The schema version is `PRAGMA user_version`; a log at another version is refused with the instruction to delete and rerun, since the log is a cache of the code plus the data.

## Blobs

Every sha reference in a payload carries a `format` from `formats.KNOWN`, which also says whether the format opens without this Python environment. Every shipped format does except `pickle`, admitted because scikit-learn has nothing else; a fit whose model is a pickle carries `portable = false`, so `SELECT ... FROM raw_fit WHERE NOT portable` names the models hostage to the environment. A save step declaring a format outside the set is refused at run. `zip` is the container for a model that is a foreign file plus a few numbers: named members with a `formats.json` manifest naming each member's format, portable iff every member is; a member outside the set, a nested zip, or a member named `formats.json` is refused. `arrow-arrays` holds floats only, so a model that is names plus floats, such as a linear model over named columns, is a `zip` with an `arrow-arrays` member and a `text/plain` member holding the JSON list of names, portable today.

| blob | format name | bytes |
|---|---|---|
| dataset rows | `parquet` | Parquet, `ts` as a nanosecond timestamp column plus one numeric, datetime or string column per field |
| predictions | `parquet` | Parquet, one column `prediction`; the row range is in the event |
| model | what the pipeline's `save` step declares | `save(model) -> bytes` and `load(bytes) -> model` as registered steps; `formats.py` ships `arrow-arrays` (a dict of arrays as an Arrow IPC file) and the temp-file round trip the Optiver example uses for `bonsai-msgpack`; a pipeline without them is refused at run |
| dirty diff | `text/x-diff` | the `git diff HEAD` text |
| declarations | not a blob: the JSON payload, re-imported by dotted path |

A blob's name is the sha256 of its bytes as stored, so identical content is written once and `fsck` can rehash and compare. The database cannot enforce a pointer into the filesystem; a missing blob is found by read or by `fsck`.

## Analytics

SQLite is the record; DuckDB is the analyst. `ATTACH 'ml_lab.sqlite' (TYPE sqlite)` queries the same views with columnar speed and hands frames to polars or pandas. Per-fold and aggregate metrics, and the fold-paired `t` on `head_to_head`, come from the views. A one-fold evaluation's paired test is row-paired and stays here: it reads both scores' `score_aggregate.series` for one window with `Ledger(root).get_blob(sha)` (`Ledger` creates an empty ledger at a root without one, where `lab run` refuses, so check the path) and `formats.arrays_load`, and their rows pair one to one when `fold_rows` match. For a metric that sums over rows, the row differences `d` give `pooled_delta = d.sum()` and an iid standard error `sqrt(len(d)) * d.std(ddof=1)`; a ratio metric or serially dependent rows want a block bootstrap of `metrics` over paired blocks, which the project owns. Nothing in the log is redesigned for analytics.

## From inception to experimentation

1. A project is a git repo exposing three module attributes, in one file or several: `dataset(ledger, *args)`, `pipelines` and `evaluations` (a list, or a function of the dataset id returning one; every pipeline is scored under each, so a scoring sweep is one list of `dataclasses.replace(base, config=..., name=...)`). Evaluation names and ids are one to one; an empty list of evaluations, or a run with no pipelines, is refused. `lab` reads them by name, so a file holding only new pipelines runs beside the project's declarations.
2. `lab ingest` appends `dataset_recorded` and stores the rows under their sha. The id is the data, so a loader fix that changes rows is a new dataset and a re-ingest of identical rows writes nothing.
3. The first `lab run` appends `evaluation_declared` with the schedule expanded once.
4. A `lab run` with work appends `run_started`, then only the fits and predictions the memo rule does not cover, then one `score_recorded` per pipeline whose predictions are new. A rerun of an unchanged tree writes nothing and prints what it reused; adding one pipeline costs only that pipeline's fits, predictions and score.
5. Read with SQL: the latest score per pipeline, its aggregate and per-fold scores, the failures. Add a `Pipeline`, `lab run`, query again.
6. Code evolves: changed code keys refit, new scores appear, old scores stand with their commit and shas.
7. Reproduce: the score names its run and predictions, the run its commit and resolution file, the fit its shas, lock and blob. Check out, sync, load, rerun, compare shas.

## Command line

```
lab ingest project/dataset.py <args>              # prints the dataset id
lab run project/experiment.py                   # newest dataset of the one source and actor; --dataset <source or id prefix>
lab run project/experiment.py ideas/agent7.py   # pipelines from both, the evaluations from one
lab run project/experiment.py --dry-run         # what a run would compute and why; writes nothing
sqlite3 -box .ml-lab/ml_lab.sqlite "..."
```

Two verbs. A module is a dotted name importable from the current directory or a `.py` path, resolved by its package so its own imports work; two `.py` paths that import under one module name are refused. The ledger root is `ML_LAB_ROOT` or `--root`; only `lab ingest` creates a ledger, and `lab run` against a root without one refuses with the resolved path. Also refused: a root that is not a directory, an ingest module without `dataset()`, `pipelines` or `evaluations` that is not a list (or, for `evaluations`, a function), and an id prefix matching several datasets, with the matches listed. With a list of evaluations each names its own dataset, so `--dataset` selects nothing and refuses an evaluation declared on another dataset. Every write carries the actor.

Reads are SQL over the views. Four to start from:

```sql
-- how each pipeline did at the first age
SELECT name, metric, fold_mean, fold_std, folds, value FROM board
WHERE evaluation_name = 'validation' AND window = '1' ORDER BY metric, fold_mean DESC;

-- every pipeline against one reference, fold by fold
SELECT name, mean_delta, delta_std, wins, folds FROM head_to_head
WHERE reference_name = 'zero' AND evaluation_name = 'validation' AND window = '1' AND metric = 'pnl'
ORDER BY mean_delta DESC;

-- one pipeline fold by fold
SELECT f.fold, f.label, f.window, f.metric, f.value FROM score_latest l
JOIN score_fold f ON f.score = l.score
WHERE l.name = 'ridge_3m' ORDER BY f.fold, f.window, f.metric;

-- what failed, and in which run
SELECT p.name, x.error, x.run, x.at FROM raw_failure x JOIN raw_pipeline p ON p.id = x.pipeline;

-- what each scored pipeline cost to fit
SELECT l.name, COUNT(*) AS fits, SUM(sf.duration_s) AS seconds FROM score_latest l
JOIN score_fit sf ON sf.score = l.score GROUP BY l.score ORDER BY seconds DESC;
```

## Later

| item | reopener |
|---|---|
| decisions: a baseline per evaluation, promote and reject on a score, a board with Pareto verdict and relative deltas, history | when reading the SQL by hand stops being how the evaluations are judged. Design kept: a score is content-addressed by its prediction ids, so a decision pins to content and new content needs a new decision; the baseline is the last promote; nothing is deleted, a retraction is a new decision. Stories S3 and S6 describe it |
| ordering rule for incomparable sets | when reading deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation. A hyperparameter picked on the metric inside `fit` is under the memo (the fit id covers the candidates in its config and the scorer's code keys when the fit imports it) but its per-fold choice and inner scores are visible only inside the model bytes; a `zip` model with a `selection` member as `arrow-arrays` records them today, and a hand-built inner window bypasses the embargo and the probe, which is the case for a tool-provided one |
| stacked blends: members fit on an inner split so the combiner learns on out-of-fold predictions | the first learned combiner wanted; today a blend with `in_sample=True` receives the members' in-sample train-range predictions as `fit`'s fourth parameter, one without receives none and sets fixed weights |
| auxiliary training rows: a second dataset a pipeline declares, which `fit` reads cut by `ts` at its train end | when a reproduction's gap is dominated by training-only data; today extra rows fork the dataset id and every evaluation, and a pipeline that must have them carries the extra file's sha in its config (so it enters the pipeline and fit ids) and guards the cut itself |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation in its own module, so `lab run` of the validation module never scores it, which is also why there is no `--evaluation` flag |
| deploy, bundle, refresh seating | when a pipeline goes to production from the log |
| materialized projections, typed payloads | when a view read passes a second, or the payload shapes stop moving |
| remote executors, agent journal, lineage export, a blob remote | when a run leaves the laptop |
