# forestry: design spec (draft 7, 2026-10-03)

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened and what was decided. People and agents write through the same five commands. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

The model is git plus code review, not a bank ledger. Datasets, pipelines, evaluations, fits, predictions and scored entries are content-addressed objects. Each evaluation has one movable ref, the baseline. Decisions are its reflog. An approval is pinned to content: new content resets review.

## Scope

In: frozen datasets, pipeline and evaluation declarations, runs that memoize fits and predictions, per-fold and aggregate scores, decisions that move a baseline, read-back views, a shell command line, analytics over the log. Out: automation that moves a baseline, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## Three systems and a boundary

| concern | owner | what the log records |
|---|---|---|
| code | git | per run: commit, dirty flag, the dirty diff as a blob; per fit: the git blob sha of every repo module imported while fitting, and the environment lock |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, diffs, each under its sha |
| facts | the event log, `forestry.sqlite` | the eight event types below |

The log never stores what a function is, only a dotted path and the blob shas git computed for the files. Reproducing a number is the join: the log says which shas and which commit, git has the content, the blob store has the bytes.

## Declarations

A pipeline is `fit(session, train_range, config) -> model` and `predict(model, session, range) -> predictions`. A scorer is `score(predictions, session, range, config) -> series`, one row per prediction, with `metrics(series) -> {name: value}` and a direction per metric declared once on the scorer; the same `metrics` runs per fold and over the concatenated folds.

`session` is the frozen dataset's columns, resident in memory, read by row range. Anything that learns from data lives inside `fit` over `train_range`. Declarations are frozen dataclasses referencing functions by dotted path, hashed by canonical serialization: a value change changes the id, a pipeline's name is a label outside the hash. No YAML, no closures. Identity is declaration only; code changes are caught by the memo rule, not by hashing files into identities.

## Schema

One table. Rows are only inserted.

```sql
CREATE TABLE event (
  seq     INTEGER PRIMARY KEY,   -- global order
  id      TEXT UNIQUE,           -- content hash for objects, ULID for runs and decisions
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
| dataset_frozen | dataset | dataset id = hash(recipe) | process, instrument, window, recipe (params, filter paths, targets), rows, blob sha |
| pipeline_declared | pipeline | pipeline id = hash(declaration) | name, fit path, predict path, config |
| evaluation_declared | evaluation | evaluation id = hash(declaration) | dataset id, scorer path, config, metric directions, cadence, the expanded folds and eval windows |
| run_started | evaluation | run id (ULID) | commit, dirty, diff sha, env lock, host facts |
| fit_computed | dataset | fit id = hash(dataset, pipeline, train range, import shas, env lock) | run id, model sha, duration |
| predictions_computed | dataset | hash(fit id, range) | blob sha |
| entry_scored | evaluation | entry id = hash(evaluation, pipeline, prediction ids) | per-fold metrics, aggregate per age |
| decision_recorded | evaluation | entry id | kind (promote, reject), why, against (baseline entry at the time, verdict, deltas) |

Fits and predictions live on the dataset stream because a scoring change reuses them across evaluations. An entry is "this pipeline, under this evaluation, from exactly these predictions"; a rerun that reproduces the same predictions writes no entry, and a code change that changes them writes a new one with no decisions.

### Views

Shipped as SQL in the same file so `sqlite3` shows them as tables. Every one takes `WHERE seq <= N` to read the log as it stood.

| view | definition |
|---|---|
| dataset, pipeline, evaluation, run, fit, prediction | one event type each, payload fields as columns |
| fold_score, aggregate_score | `entry_scored` unpacked one row per (fold, age, metric) and per (age, metric) |
| latest_entry | per (evaluation, pipeline), the newest entry |
| baseline | per evaluation, the entry of the last promote decision |
| status | per entry: baseline if head, superseded if promoted earlier, rejected if its last decision is reject, else scored |
| board | latest_entry beside baseline with both aggregates; verdict and relative deltas from the metric directions |
| history | promote decisions in order |

The payload is JSON so a shape change is a new payload version and an edited view, not a migration. Materialize a view only when a read is measured past a second; a `reproject` command then rebuilds it and nothing else writes it.

## Rules

Memo: a fit is reused when one exists for (dataset, pipeline, train range) whose recorded import shas and environment lock all match the current tree. Dirty files hash like any other; the run records the diff. A dependency bump changes the lock and refits. Predictions are reused per (fit, range).

Review: a decision is made on one entry and carries the baseline entry, verdict and deltas it saw. The baseline is the last promote. New content for a pipeline is a new entry with no decisions, so the board asks for review again. Nothing is deleted; a refutation is a new decision.

Comparability: an entry compares only against the baseline of its own evaluation. A scorer or schedule change is a new evaluation.

Embargo: no fit's train range ends after its fold's eval start minus the embargo. The folds are in the evaluation payload, written once.

Day one, because rows written wrong cannot be repaired: ids that merge across hosts (content hashes and ULIDs, no serial counters); the blob store writes to a temp file, fsyncs and renames; every fit carries its import shas and env lock. Two logs from two hosts merge by `INSERT OR IGNORE` on id, decisions on one host.

## Analytics

SQLite is the record; DuckDB is the analyst. `ATTACH 'forestry.sqlite' (TYPE sqlite)` queries the same views with columnar speed and hands frames to polars or pandas. Per-fold and aggregate metrics come from the views; per-row work (paired tests on pnl) loads prediction blobs by sha. Nothing in the log is redesigned for analytics.

## From inception to experimentation

1. A project is three modules in a git repo: how to read the raw data, the steps, the declarations (`pipelines` and `evaluation(dataset)`).
2. `fy freeze` appends `dataset_frozen` and stores the rows under their sha. Same recipe, same id, no write.
3. The first `fy run` appends `evaluation_declared` with the schedule expanded once.
4. Every `fy run` appends `run_started`, then only the fits and predictions the memo rule does not cover, then one `entry_scored` per pipeline whose predictions are new.
5. `fy decide <pipeline> --kind promote --why "incumbent"` appends the first decision; the baseline view resolves to that entry.
6. Add a `Pipeline`, commit, `fy run`, `fy board`, `fy decide`. `fy history` is the reflog; `fy board <pipeline>` is one pipeline with every entry, run and decision.
7. Code evolves: changed import shas refit, new entries appear as scored, old entries and their decisions stand with their commit and shas.
8. Reproduce: the entry names its run and predictions, the run its commit and lock, the fit its shas and blob. Check out, sync, load, rerun, compare shas.

## Command line

```
DS=$(fy freeze project.capture:freeze <args>)
C="project.declarations --dataset $DS"
fy run $C
fy decide $C <pipeline> --kind promote --why "incumbent"
fy board $C
fy board $C <pipeline>
fy decide $C <pipeline> --kind promote|reject --why "..."
fy history $C
```

Five verbs. Pipelines are named by name or id prefix. The ledger root is `FORESTRY_ROOT` or `--root`. Every write carries the actor.

## Later

| item | reopener |
|---|---|
| ordering rule for incomparable sets | when reading the board's deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation |
| deploy, bundle, refresh seating | when a pipeline goes to production from the log |
| materialized projections, typed payloads | when a view read passes a second, or the payload shapes stop moving |
| remote executors, agent journal, lineage export, a blob remote | when a run leaves the laptop |
