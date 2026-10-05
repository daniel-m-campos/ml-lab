# ml-lab: the log

A local-first record of experimentation on frozen datasets whose rows are in a fixed order, time series among them. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. People and agents write through two commands and read with SQL. Datasets, pipelines, evaluations, fits, predictions and scores are content-addressed, so the same work is never done twice and every number names the code, data and environment that produced it.

This file holds what the code cannot say for itself: the event and view tables, the blob formats, the designs declined with their reasons, and what is deferred. How each function behaves, what is refused and with which message, lives in the module docstrings and `tests/test_ledger_story.py`.

## Three systems and a boundary

| concern | owner | what the log records |
|---|---|---|
| code | git | per run: commit, dirty flag, the dirty diff as a blob; per fit: the git blob sha of every repo module the pipeline's functions import, and the environment lock |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, diffs, each under its sha, in a format that opens without this Python environment |
| facts | the event log, `ml_lab.sqlite` | the nine event types below |

The log never stores what a function is, only a dotted path and the blob shas git computed for the files. Reproducing a number is the join: the log says which shas and which commit, git has the content, the blob store has the bytes.

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
| dataset_recorded | dataset | dataset id = hash(`ts` bytes, column names, dtypes, values (numeric and datetime by bytes, strings by value, other objects refused), targets with reveal lags) | source, window, recipe (params, filter paths, targets with reveal lags in seconds or as a count of dates) as provenance, rows, blob sha. A dataset with no rows, a column named `ts` (the clock's name), a missing target, a target that is not numeric or boolean (targets are masked with NaN), a reveal lag naming a non-target, a reveal lag without a clock, or a lag that is negative or neither a timedelta nor an int is refused. A source is a label: `--dataset <source>` resolves to its newest dataset only while one actor wrote under it, and two actors under one source are refused with both ids |
| pipeline_declared | pipeline | event id = hash(pipeline id, name); key = pipeline id = hash(declaration) | name and the declaration: fit, predict, save, load, features and postprocess as dotted paths, format, config, postprocess config, member declarations for a blend; the views show the newest name per id, and one run declaring one id under two names is refused |
| evaluation_declared | evaluation | event id = hash(evaluation id, name); key = evaluation id = hash(declaration) | name, dataset id, split, the dotted path of the scorer's `score`, config, metric directions as first declared (comparisons read each score's), the expanded folds (label, train segments, test range); a split that yields other folds under an unchanged declaration is refused |
| run_started | evaluation | run id (ULID) | commit, dirty (tracked changes or untracked files), diff sha (one `git diff HEAD` over a temporary index holding untracked files; one over 1 MiB is named, not diffed), resolution file sha (`uv.lock` or `requirements*.txt`), host facts, the run's pipelines (id, name), and `editable`: per editable install the pipelines' functions reach, its git root's commit, dirty flag and diff |
| features_computed | dataset | feature id = hash(dataset, one features function's path, its code keys, its env lock sha) | pipeline id, run id, column names, the columns id (ordered names, dtypes and values, strings by value), import shas and code keys, the probe rows, duration, the columns as a Parquet blob; a tuple of functions is one event per function |
| fit_computed | dataset | fit id = hash(dataset, fit, save and load paths, format and config, train segments, their code keys, env lock sha, the feature columns id when the pipeline has features: the hash of its features functions' column ids in order) | run id, label, model sha, format and whether it is portable, import shas, env lock sha, duration; a blend's fit also hashes and records its members' fit ids. A feature set, fit, prediction or postprocess whose module files changed on disk between the run's start and its write is refused, so a sha is never recorded for code that did not run |
| predictions_computed | dataset | raw: hash(fit id, range, predict path and its code keys); postprocessed: hash(raw id, postprocess path, postprocess config and its code keys) | blob sha and format, fold index; a postprocessed one names its raw id, postprocess path and config; a blend's names its members' prediction ids, and, only when the blend declares `in_sample=True`, its members' in-sample predictions over the train segments are predictions too, under the same fold index and told apart by their range |
| score_recorded | evaluation | score id = hash(evaluation, pipeline, prediction ids, the code keys of the scorer's `score` and `metrics`) | per-fold metrics with the fold label, the aggregate metrics over the concatenated folds (a non-finite metric is stored as null), the scorer's metric directions and import shas, and the scorer's series as one `arrow-arrays` blob with the fold row counts, its columns named by `Scorer(columns=...)` or `0`, `1`, ..., so any uncertainty method can be run later; a `columns` count the series disagrees with fails the pipeline as `pipeline_failed` |
| pipeline_failed | evaluation | pipeline id | run id, error, traceback sha |

Fits and predictions live on the dataset stream because a scoring change reuses them across evaluations. A score is "this pipeline, under this evaluation, from exactly these predictions"; a rerun that reproduces the same predictions writes no score, and a code change that changes them writes a new one.


### Views

Shipped as SQL in the same file so `sqlite3` shows them as tables; a connect drops and recreates them in one write transaction, so concurrent `lab run` processes never see a half-built set. The names say the layer: `board` and `head_to_head` are for reading and carry no prefix; `raw_` is one event type as columns, for joins; `score_` is a building block over scores. `.tables raw%` lists a layer. The `raw_` views carry `seq`, so `WHERE seq <= N` reads the log as it stood, except `raw_pipeline` and `raw_evaluation` (latest name), `raw_failure` (dropped once a newer score exists) and `score_latest`, which read the whole log.

| view | definition |
|---|---|
| board | how each pipeline did: per (evaluation, pipeline, metric) the latest score's `fold_mean`, `fold_std`, `folds` and `value`, in that order, with the dataset source and the evaluation and pipeline names first and the ids last; `score_latest` joined to `score_aggregate` |
| head_to_head | is A better than B: per (evaluation, metric) and ordered pair of latest pipelines, with the evaluation name and dataset source: `folds`, those where both metrics are non-null; `mean_delta`, the mean per-fold difference (the difference of the pair's `fold_mean`); `delta_std`, its sample std, NULL at one fold; `t`, fold-paired, folds - 1 degrees of freedom, independent folds assumed, NULL at one fold or zero spread; `wins`, the folds the first wins by the score's direction; `pooled_delta`, the difference of pooled `value`. A comparison's noise is `delta_std`, not `board`'s `fold_std`; at one fold, pair the series (Analytics) |
| raw_dataset, raw_pipeline, raw_evaluation, raw_run, raw_feature, raw_fit, raw_prediction, raw_score, raw_failure | one event type each, payload fields as columns; `raw_evaluation.config` is canonical, so a cost sweep's rows differ without names; `raw_run` carries `editable` and `pipelines`; `raw_failure` carries the run and error until a newer score for its evaluation and pipeline |
| score_fold, score_aggregate | `score_recorded` unpacked per (fold, metric) and per metric. `value` is `metrics` over the concatenated folds, `fold_mean` the mean of fold values; a null fold (a non-finite metric) is left out of `folds`, `fold_mean` and `fold_std`. For a metric that does not add over rows (AUC, a rank metric, a correlation, a Sharpe) they differ and can disagree in sign; a constant predictor whose constant moves across folds scores a pooled AUC off 0.5. So `board` lists `fold_mean` first. The aggregate row also carries `folds`, `fold_std`, the series blob sha and its `fold_rows` |
| score_latest | per (evaluation, pipeline), the newest score, with the pipeline and evaluation names and the dataset's id and source joined on. A label is not an id: a declaration change under one name lists the name once per id, and the log cannot say which declaration the code holds (a rerun that reuses everything writes nothing), so a board filters on the id `lab run` prints |
| score_fit | one row per (score, fit): the fits a score stands on, including a blend's members, with the fit's pipeline, label and seconds |

The payload is JSON so a shape change is a new payload version and an edited view, not a migration. Materialize a view only when a read is measured past a second.

## Blobs

Every sha reference in a payload carries a `format` from `formats.KNOWN`, which also says whether the format opens without this Python environment. Every shipped format does except `pickle`, admitted because scikit-learn has nothing else; a fit whose model is a pickle carries `portable = false`, so `SELECT ... FROM raw_fit WHERE NOT portable` names the models hostage to the environment. A pipeline declaring a `format` outside the set is refused at run. `zip` is the container for a model that is a foreign file plus a few numbers: named members with a `formats.json` manifest naming each member's format, portable iff every member is; a member outside the set, a nested zip, or a member named `formats.json` is refused. `arrow-arrays` holds floats only, so a model that is names plus floats, such as a linear model over named columns, is a `zip` with an `arrow-arrays` member and a `text/plain` member holding the JSON list of names, portable today.

| blob | format name | bytes |
|---|---|---|
| dataset rows | `parquet` | Parquet, `ts` as a nanosecond timestamp column plus one numeric, datetime or string column per field |
| predictions | `parquet` | Parquet, one column `prediction`, or `prediction_0` to `prediction_{k-1}`; the row range is in the event |
| model | the pipeline's `format` | `save(model) -> bytes` and `load(bytes) -> model` as module-level functions; `formats.py` ships `arrow-arrays` (a dict of arrays as an Arrow IPC file) and the temp-file round trip for a library that saves to a path, `bonsai-msgpack` among them; a pipeline without them is refused at run |
| dirty diff | `text/x-diff` | the `git diff HEAD` text |
| declarations | not a blob: the JSON payload, re-imported by dotted path |

A blob's name is the sha256 of its bytes as stored, so identical content is written once and `fsck` can rehash and compare. The database cannot enforce a pointer into the filesystem; a missing blob is found by read or by `fsck`.

## Declined

Designs proposed by a study and declined, with the reason, so a round does not re-argue them. Each stands until its reason stops holding.

| proposal | kept instead | reason |
|---|---|---|
| hash source files into identities | identity is declaration only; a module's code key (syntax tree without comments, layout or docstrings) is under the memo, its git blob sha recorded beside it | a formatter pass or a comment edit must keep every id; a code edit must refit, which the memo does |
| scorer `metrics` and `directions` in the evaluation id; scorer path out of it so two studies pair | the evaluation id hashes the scorer's path and config; `metrics` and `directions` are code under the memo; each score carries the directions it was scored under; two studies share an evaluation by importing one scorer module | a scorer edit is a new score under the same evaluation, not a new evaluation; a copy at another path is another evaluation |
| `metrics(series, config)` | `metrics(series)` on the stored blob alone; a constant column (a holding period) carries what it needs; per-row config such as costs stays in the scorer body | the series must re-score without the scorer's code or config |
| strings in `arrow-arrays` | a `zip` with an `arrow-arrays` member and a `text/plain` member holding the JSON names | one dtype path in the format that also holds every score series |
| mask labels at the train end for a reveal lag | a fold that trains on a row whose label is not known at its test range's start is refused before declaration, naming the embargo that clears it | masking measures from the wrong instant and puts NaN labels into every lagged-target fit |
| a per-element ulp tolerance in the probes | within `1e-9` of the range's largest value (1024 ulps for narrower floats), NaNs equal | honest BLAS and layout differences stay under 3e-12 of scale, the smallest measured leak is 1e-7, and an ulp rule is unsound near zero |
| refit cadence as a pipeline property | an online variant that refits every N days is a split with growing train ranges, beside the house evaluation, paired at the series level when the validation ranges concatenate to the same rows | a schedule is part of the evaluation; scores compare only within one |
| refuse when the editable tool is edited mid-run | sources read once per process; one `lab run` fits under one lock; `run_started.editable` records the tool's commit and diff | aborting every agent's run on every tool edit |
| `--evaluation` flag | the test evaluation lives in its own module, so `lab run` of the validation module never scores it | a flag is one typo from scoring the test set |
| `git read-tree HEAD` for the dirty diff index | the index is copied to a temp file and untracked files added with intent | `read-tree` drops staged, uncommitted files |
| fold slices as independent samples | several folds over one train range share one fit; slices of one day are serially dependent, their spread is a floor on the noise | `fold_std` over slices understates it |
| several validation windows per fold (horizons) | one validation range per fold; a later horizon is another evaluation on the same cutoffs, sharing every fit through the memo; finer decay is read from the stored series | two levels where one does; the board had to be filtered by window before it could be averaged |
| a declared `Pipeline(outputs=...)` width | `predict` returns `(rows,)` or `(rows, k)`; the width is a property of the step | the declaration would say what the code already says and be checked only when the code runs |

## Analytics

SQLite is the record; DuckDB is the analyst. `ATTACH 'ml_lab.sqlite' (TYPE sqlite)` queries the same views with columnar speed and hands frames to polars or pandas. Per-fold and aggregate metrics, and the fold-paired `t` on `head_to_head`, come from the views. A one-fold evaluation's paired test is row-paired and stays here: it reads both scores' `score_aggregate.series` with `Ledger(root).get_blob(sha)` (`Ledger` creates an empty ledger at a root without one, where `lab run` refuses, so check the path) and `formats.arrays_load`, and their rows pair one to one when `fold_rows` match. For a metric that sums over rows, the row differences `d` give `pooled_delta = d.sum()` and an iid standard error `sqrt(len(d)) * d.std(ddof=1)`; a ratio metric or serially dependent rows want a block bootstrap of `metrics` over paired blocks, which the project owns. Nothing in the log is redesigned for analytics.

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
-- how each pipeline did
SELECT name, metric, fold_mean, fold_std, folds, value FROM board
WHERE evaluation_name = 'validation' ORDER BY metric, fold_mean DESC;

-- every pipeline against one reference, fold by fold
SELECT name, mean_delta, delta_std, wins, folds FROM head_to_head
WHERE reference_name = 'zero' AND evaluation_name = 'validation' AND metric = 'pnl'
ORDER BY mean_delta DESC;

-- one pipeline fold by fold
SELECT f.fold, f.label, f.metric, f.value FROM score_latest l
JOIN score_fold f ON f.score = l.score
WHERE l.name = 'ridge_3m' ORDER BY f.fold, f.metric;

-- what failed, and in which run
SELECT p.name, x.error, x.run, x.at FROM raw_failure x JOIN raw_pipeline p ON p.id = x.pipeline;

-- what each scored pipeline cost to fit
SELECT l.name, COUNT(*) AS fits, SUM(sf.duration_s) AS seconds FROM score_latest l
JOIN score_fit sf ON sf.score = l.score GROUP BY l.score ORDER BY seconds DESC;
```

## Later

| item | reopener |
|---|---|
| decisions: a baseline per evaluation, promote and reject on a score, a board with Pareto verdict and relative deltas, history | when reading the SQL by hand stops being how the evaluations are judged. Design kept: a score is content-addressed by its prediction ids, so a decision pins to content and new content needs a new decision; the baseline is the last promote; nothing is deleted, a retraction is a new decision. |
| ordering rule for incomparable sets | when reading deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation. A hyperparameter picked on the metric inside `fit` is under the memo (the fit id covers the candidates in its config and the scorer's code keys when the fit imports it) but its per-fold choice and inner scores are visible only inside the model bytes; a `zip` model with a `selection` member as `arrow-arrays` records them today, and a hand-built inner validation range bypasses the embargo and the probe, which is the case for a tool-provided one |
| stacked blends: members fit on an inner split so the combiner learns on out-of-fold predictions | the first learned combiner wanted; today a blend with `in_sample=True` receives the members' in-sample train-range predictions as `fit`'s fourth parameter, one without receives none and sets fixed weights |
| auxiliary training rows: a second dataset a pipeline declares, which `fit` reads cut by `ts` at its train end | when a reproduction's gap is dominated by training-only data; today extra rows fork the dataset id and every evaluation, and a pipeline that must have them carries the extra file's sha in its config (so it enters the pipeline and fit ids) and guards the cut itself |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation in its own module, so `lab run` of the validation module never scores it, which is also why there is no `--evaluation` flag |
| deploy, bundle, refresh seating | when a pipeline goes to production from the log |
| materialized projections, typed payloads | when a view read passes a second, or the payload shapes stop moving |
| remote executors, agent journal, lineage export, a blob remote | when a run leaves the laptop |
