# forestry: design spec (draft 8, 2026-10-03)

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. People and agents write through two commands and read with SQL. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

Datasets, pipelines, evaluations, fits, predictions and scores are content-addressed objects, so the same work is never done twice and every number names the code, data and environment that produced it. Decisions over scores (a baseline, promote and reject, a board) are designed but deferred to Later: the first job is to analyze the evaluations by hand and let that refine the schema.

## Scope

In: frozen datasets, pipeline and evaluation declarations, runs that memoize fits and predictions, per-fold and aggregate scores, SQL views, a two-verb command line, analytics over the log. Out, for now: decisions and a baseline (Later), automation that moves one, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## Three systems and a boundary

| concern | owner | what the log records |
|---|---|---|
| code | git | per run: commit, dirty flag, the dirty diff as a blob; per fit: the git blob sha of every repo module imported while fitting, and the environment lock |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, diffs, each under its sha, in a format that opens without this Python environment |
| facts | the event log, `forestry.sqlite` | the seven event types below |

The log never stores what a function is, only a dotted path and the blob shas git computed for the files. Reproducing a number is the join: the log says which shas and which commit, git has the content, the blob store has the bytes.

## Declarations

An evaluation is a dataset, a split, a scorer and its config. The split is a frozen dataclass with `folds(session)`: `WalkForward` (cutoffs every N rows, a window after each, optional named horizons), `BlockedKFold` and `Holdout` work on any session by row count; `CalendarWalkForward` (monthly cutoffs on the session clock, horizons in months, an embargo in seconds) needs `ts`. A project may declare its own. A fold trains on contiguous segments and scores named windows. A pipeline is `fit(session, train_segments, config) -> model`, `predict(model, session, range) -> predictions`, and `save(model) -> bytes` with `load(bytes) -> model` naming a format that opens without Python. A scorer is `score(predictions, session, range, config) -> series`, one row per prediction, with `metrics(series) -> {name: value}` and a direction per metric declared once on the scorer; the same `metrics` runs per fold and over the concatenated folds.

`session` is the frozen dataset's columns, resident in memory, read by row range. Anything that learns from data lives inside `fit` over `train_range`. Declarations are frozen dataclasses referencing functions by dotted path, hashed by canonical serialization: a value change changes the id, a pipeline's name is a label outside the hash. No YAML, no closures. Identity is declaration only; code changes are caught by the memo rule, not by hashing files into identities.

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
| dataset_recorded | dataset | dataset id = hash(recipe) | source, window, recipe (params, filter paths, targets), rows, blob sha |
| pipeline_declared | pipeline | pipeline id = hash(declaration) | name, fit path, predict path, config |
| evaluation_declared | evaluation | evaluation id = hash(declaration) | dataset id, split, scorer path, config, metric directions, the expanded folds (label, train segments, named windows) |
| run_started | evaluation | run id (ULID) | commit, dirty, diff sha, resolution file sha (`uv.lock` or `requirements*.txt` if present), host facts |
| fit_computed | dataset | fit id = hash(dataset, pipeline, train range, import shas, env lock sha) | run id, cutoff, model sha and format, import shas, env lock sha, duration |
| predictions_computed | dataset | hash(fit id, range) | blob sha and format |
| score_recorded | evaluation | score id = hash(evaluation, pipeline, prediction ids) | per-fold metrics, aggregate per age |
| pipeline_failed | evaluation | pipeline id | run id, error, traceback sha |

Fits and predictions live on the dataset stream because a scoring change reuses them across evaluations. A score is "this pipeline, under this evaluation, from exactly these predictions"; a rerun that reproduces the same predictions writes no score, and a code change that changes them writes a new one.

### Views

Shipped as SQL in the same file so `sqlite3` shows them as tables. Every view takes `WHERE seq <= N` to read the log as it stood.

| view | definition |
|---|---|
| dataset, pipeline, evaluation, run, fit, prediction, score | one event type each, payload fields as columns |
| fold_score, aggregate_score | `score_recorded` unpacked one row per (fold, window, metric) and per (window, metric) |
| latest_score | per (evaluation, pipeline), the newest score |
| failure | `pipeline_failed` with run and error |

The payload is JSON so a shape change is a new payload version and an edited view, not a migration. Materialize a view only when a read is measured past a second.

## Rules

Memo: a fit is reused when one exists for (dataset, pipeline, train range) whose recorded import shas and environment lock all match the current tree. Dirty files hash like any other; the run records the diff. Predictions are reused per (fit, range).

The environment lock is a text blob, one `name==version` line per installed distribution the pipeline's import closure reaches, with their requirements, plus the Python version. Installing an unrelated package changes nothing; bumping a dependency refits the pipelines that import it, and two lock blobs diff as text. An editable install's version carries a hash of its source files. Platform is recorded on the run, not in the lock, so a GPU box reuses a laptop's fits.

The import closure is every module under the repo root reachable from the step modules' top-level imports, with `.venv` excluded. `experiment.py` is in it only if a step imports it, so editing a config there changes the pipeline id, not the fit shas. A repo module or a distribution imported lazily inside a step is invisible to the closure; after the first fit the run sees it loaded, refuses, and names it, because the memo would otherwise have missed its edits or its version bumps.

Comparability: scores compare only within one evaluation. A scorer or schedule change is a new evaluation.

Faults: every blob is written before the event that names it, and each fit is its own event, so a killed process loses at most the fit in flight and a rerun resumes from the last recorded one with the same ids. A pipeline that raises is recorded as `pipeline_failed` with its traceback, the other pipelines continue, and `fy run` exits 1 naming it. Failures are not memoized: a rerun retries. The `failure` view holds the error until a newer score exists.

Splits: a fit's train segments never overlap its fold's windows, and the embargo the split declares separates them. The folds are in the evaluation payload, written once; `session.ts` is optional and only `CalendarWalkForward` needs it.

Day one, because rows written wrong cannot be repaired: ids that merge across hosts (content hashes and ULIDs, no serial counters); the blob store writes to a temp file, fsyncs and renames; every fit carries its import shas and env lock. Two logs from two hosts merge by `INSERT OR IGNORE` on id. The schema version is `PRAGMA user_version`; a log at another version is refused with the instruction to delete and rerun, since the log is a cache of the code plus the data.

## Blobs

Every blob opens without this Python environment, and every sha reference in a payload carries a `format`. Pickle is never written.

| blob | format name | bytes |
|---|---|---|
| dataset rows | `parquet` | Parquet, `ts` as a timestamp column plus one float64 column per field |
| predictions | `parquet` | Parquet, one column `prediction`; the row range is in the event |
| model | what the pipeline's `save` step declares | `save(model) -> bytes` and `load(bytes) -> model` as registered steps; `formats.py` ships `arrow-arrays` (a dict of arrays as an Arrow IPC file) and the temp-file round trip the Optiver example uses for `bonsai-msgpack`; a pipeline without them is refused at run |
| dirty diff | `text/x-diff` | the `git diff HEAD` text |
| declarations | not a blob: the JSON payload, re-imported by dotted path |

A blob's name is the sha256 of its bytes as stored, so identical content is written once and `fsck` can rehash and compare. The database cannot enforce a pointer into the filesystem; a missing blob is found by read or by `fsck`.

## Analytics

SQLite is the record; DuckDB is the analyst. `ATTACH 'forestry.sqlite' (TYPE sqlite)` queries the same views with columnar speed and hands frames to polars or pandas. Per-fold and aggregate metrics come from the views; per-row work (paired tests on pnl) loads prediction blobs by sha. Nothing in the log is redesigned for analytics.

## From inception to experimentation

1. A project is a git repo exposing three module attributes, in one file or several: `dataset(ledger, *args)`, `pipelines` and `evaluation` (a value or a function of the dataset id). `fy` reads them by name, so a file holding only new pipelines runs beside the project's declarations.
2. `fy ingest` appends `dataset_recorded` and stores the rows under their sha. Same recipe, same id, no write.
3. The first `fy run` appends `evaluation_declared` with the schedule expanded once.
4. A `fy run` with work appends `run_started`, then only the fits and predictions the memo rule does not cover, then one `score_recorded` per pipeline whose predictions are new. A rerun of an unchanged tree writes nothing and prints what it reused; adding one pipeline costs only that pipeline's fits, predictions and score.
5. Read with SQL: the latest score per pipeline, its aggregate and per-fold scores, the failures. Add a `Pipeline`, `fy run`, query again.
6. Code evolves: changed import shas refit, new scores appear, old scores stand with their commit and shas.
7. Reproduce: the score names its run and predictions, the run its commit and resolution file, the fit its shas, lock and blob. Check out, sync, load, rerun, compare shas.

## Command line

```
fy ingest project/dataset.py <args>              # prints the dataset id
fy run project/experiment.py                   # newest dataset; --dataset <id prefix> to pick
fy run project/experiment.py ideas/agent7.py   # pipelines from both, the evaluation from one
sqlite3 -box .forestry/forestry.sqlite "..."
```

Two verbs. A module is a dotted name importable from the current directory or a `.py` path, resolved by its package so its own imports work. The ledger root is `FORESTRY_ROOT` or `--root`. Every write carries the actor.

Reads are SQL over the views. Three to start from:

```sql
-- latest aggregate scores per pipeline at the first age
SELECT p.name, s.metric, s.value FROM latest_score l
JOIN aggregate_score s ON s.score = l.score JOIN pipeline p ON p.id = l.pipeline
WHERE s.window = '1' ORDER BY s.metric, s.value DESC;

-- one pipeline fold by fold
SELECT f.fold, f.label, f.window, f.metric, f.value FROM latest_score l
JOIN fold_score f ON f.score = l.score JOIN pipeline p ON p.id = l.pipeline
WHERE p.name = 'ridge_3m' ORDER BY f.fold, f.window, f.metric;

-- what failed, and in which run
SELECT p.name, x.error, x.run, x.at FROM failure x JOIN pipeline p ON p.id = x.pipeline;
```

## Later

| item | reopener |
|---|---|
| decisions: a baseline per evaluation, promote and reject on a score, a board with Pareto verdict and relative deltas, history | when reading the SQL by hand stops being how the evaluations are judged. Design kept: a score is content-addressed by its prediction ids, so a decision pins to content and new content needs a new decision; the baseline is the last promote; nothing is deleted, a retraction is a new decision. Stories S3 and S6 describe it |
| ordering rule for incomparable sets | when reading deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation |
| deploy, bundle, refresh seating | when a pipeline goes to production from the log |
| materialized projections, typed payloads | when a view read passes a second, or the payload shapes stop moving |
| remote executors, agent journal, lineage export, a blob remote | when a run leaves the laptop |
