# forestry: design spec (draft 6, 2026-10-03)

A local-first ledger for experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, a relational ledger owns what happened and what was decided. Nothing in the ledger is updated in place. People and agents write to the same ledger through the same five commands. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

## Scope

In: frozen datasets, pipeline and evaluation declarations, runs that memoize fits and predictions, per-fold and aggregate scores, decisions that move a baseline, read-back views, a shell command line. Out: automation that moves a baseline, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## Three systems and a boundary

| concern | owner | what the ledger stores |
|---|---|---|
| code | git | on every run: commit, dirty flag, environment lock, host |
| bytes | the blob store, `blobs/sha256/` | dataset rows, models, predictions, each under its sha |
| facts | the ledger, `forestry.sqlite` | accounts and entries, below |

The ledger never stores what a function is, only how to find it: a dotted path inside a declaration. It never stores a file hash of code; it asks git. The three declarations (dataset recipe, pipeline, evaluation) are opaque documents whose canonical serialization is their identity. Everything else is a column.

## Declarations

A pipeline is `fit(session, train_range, config) -> model` and `predict(model, session, range) -> predictions`. A scorer is `score(predictions, session, range, config) -> series`, one row per prediction, plus `metrics(series) -> {name: value}` and a direction per metric, declared once on the scorer. The same `metrics` runs per fold and over the concatenated folds, so the aggregate needs no second definition.

`session` is the frozen dataset's columns, resident in memory, read by row range; it is not any library's object. Anything that learns from data lives inside `fit` over `train_range`: features, binning, scaling, the training window. Declarations are frozen dataclasses referencing step and scorer functions by dotted path. Formatting changes nothing, a value change changes the id, a pipeline's name is a label outside the hash. No YAML, no closures.

Identity is declaration only. A behaviour change in code is a declared change: a version field in the config, or a new step name. The memo rule below catches the undeclared ones.

## Schema

Accounts are set up once and content-addressed. Entries are posted by runs.

### Accounts

| table | key | columns |
|---|---|---|
| dataset | dataset_id = hash(recipe) | process, instrument, window_start, window_end, rows, blob_sha, recipe (document: params, filter paths, targets) |
| pipeline | pipeline_id = hash(declaration) | name, declaration (document: fit path, predict path, config) |
| evaluation | evaluation_id = hash(declaration) | dataset_id FK, first_cutoff, every_months, eval_months, embargo_seconds, min_folds, declaration (document: scorer path, config) |
| evaluation_metric | (evaluation_id, metric) | direction |
| fold | (evaluation_id, fold_index) | cutoff, train_start, train_end |
| eval_window | (evaluation_id, fold_index, age) | range_start, range_end |

### Entries

| table | key | columns |
|---|---|---|
| host | host_id | hostname, system, machine, cpu_count, cgroup_cpu_max, python |
| run | run_id | evaluation_id FK, git_commit, git_dirty, env_lock, host_id FK, at |
| fit | fit_id | dataset_id FK, pipeline_id FK, train_start, train_end, git_commit, duration_s, model_sha |
| prediction | prediction_id | fit_id FK, range_start, range_end, blob_sha |
| entry | (run_id, pipeline_id) | the transaction: this pipeline was scored in this run |
| fold_score | (run_id, pipeline_id, fold_index, age, metric) | value |
| aggregate_score | (run_id, pipeline_id, age, metric) | value |
| decision | decision_id (serial) | run_id, pipeline_id (FK entry), kind (promote, reject), why, against_run_id, against_pipeline_id, verdict, at |
| decision_delta | (decision_id, metric) | relative_delta |

### Views

| view | definition |
|---|---|
| latest_entry | per (evaluation, pipeline), the entry from the newest run |
| baseline | per evaluation, the entry of the last promote decision |
| status | per entry: baseline, superseded, rejected, scored, from its decisions |
| comparison | latest_entry aggregates against the baseline's, with the direction from evaluation_metric: verdict and relative delta per metric |

The board is `comparison`. Foreign keys are enforced. The only non-atomic columns are the three declaration documents.

## Rules

Memo: a fit is reused for (dataset, pipeline, train range) unless git reports that the pipeline's step files changed between the fit's commit and HEAD; then it is fit again at HEAD as a new fit row. Predictions are reused per (fit, range). A dirty worktree is recorded and warned about, not refused.

Comparability: an entry compares only against the baseline of its own evaluation. A scorer or schedule change is a new evaluation, so old entries stay readable and never compare across.

Decisions: a baseline moves only through a promote decision. A decision is made on one entry and carries the baseline entry, verdict and deltas it saw. Reject closes an entry. Nothing is deleted; a refutation is a new decision.

Embargo: no fit's train range ends after its fold's eval start minus the embargo. The fold table is written once and is the proof.

## From inception to experimentation

1. A project is three modules in a git repo: how to read the raw data, the steps, the declarations (`pipelines` and `evaluation(dataset)`).
2. `fy freeze` opens the dataset account: load, filter, check targets, store the rows under their sha, write one row. Same recipe, same id, no write.
3. The first `fy run` opens the evaluation account: the evaluation row, its metrics and directions, the schedule expanded once into folds and eval windows.
4. Every `fy run` posts a run (commit, dirty, env, host) and one entry per declared pipeline, fitting and predicting only what the memo rule does not cover.
5. `fy decide <pipeline> --kind promote --why "incumbent"` seats the first baseline.
6. Add a `Pipeline` to the declarations, commit, `fy run`. `fy board` reads the comparison view. `fy decide` records promote or reject with a reason; `fy history` is the chain of promotes; `fy board <pipeline>` is one pipeline in full.
7. Code evolves: git names the step files that moved, affected pipelines refit in a new run, old entries and their decisions stand, pointing at their commit.
8. Any score joins to its run, commit and environment, and to the fit's dataset blob and train range. Check out, load, rerun.

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

Five verbs. Pipelines are named by name or id prefix. The ledger root is `FORESTRY_ROOT` or `--root`.

## Later

Each item returns only with a reason written here first.

| item | reopener |
|---|---|
| ordering rule for incomparable sets | when reading the board's deltas by hand costs more than a rule hides; a band makes dominance intransitive, so a rule needs a seating story |
| staged scoring with recorded gates | when a full simulation is expensive enough that screening must be recorded; until then two evaluations do it |
| execution config grids, tune steps, heads | when one fit must be scored under several settings in one evaluation; today a different setting is a different evaluation or pipeline |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation whose schedule covers it |
| deploy, bundle, refresh seating | when a pipeline goes to production from the ledger |
| remote executors, agent journal, lineage events, a blob remote | when a run leaves the laptop |
