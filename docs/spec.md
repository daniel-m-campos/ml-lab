# forestry: design spec (draft 5, 2026-10-03)

A local-first ledger for experimentation on frozen, time-ordered datasets. It records what was tried, under which evaluation, with what scores, and which recorded decision moved the baseline. People and agents write to the same ledger. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

Drafts 4 and 5 trim draft 3 to record keeping. Staged funnels, gates, ordering rules, tune steps, heads, the sealed window, deployment bundles, campaign contracts, and then the capture layer, candidate, comparison and baseline tables were built, found to be more surface than record, and removed; the first group is listed under "Later" with what would bring each back, the second became derived views.

## Scope

In: dataset layers and hashes, pipeline and evaluation declarations, memoized fits and predictions, per-fold scores and aggregates, comparisons against a baseline, recorded decisions, read-back views, a shell command line. Out: automation that moves a baseline, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## The pattern

An experiment is a triple on three independent, content-addressed axes. Nothing on one axis references another; the harness composes them.

| axis | holds | changes when |
|---|---|---|
| dataset | process, params, instrument, window, filters, targets | new data or a labeling idea |
| pipeline | `fit` and `predict` steps and their config, including the training window | a modeling idea |
| evaluation | dataset id, scorer and its config, first cutoff, cadence, eval window, ages, embargo, min_folds | a validation or scoring idea |

Two memo points keep the cross product from materializing. A fit is keyed by (dataset, pipeline, train_range). Predictions are keyed by (fit, eval_range). Scores derive from predictions, the session's columns over the eval range and the scorer's config, so a scoring change refits nothing.

A pipeline under an evaluation has one score row. Comparability is "same evaluation id"; a decision can only be recorded on a pipeline scored under that evaluation.

## Contracts

```
fit(session, train_range, config) -> model
predict(model, session, range) -> predictions
score(predictions, session, range, config) -> series      # one row per prediction
metrics(series) -> {name: value}                          # declared on the scorer
```

`session` is the frozen dataset's raw columns, resident in memory; it is not any library's object. Anything that learns from data lives inside `fit` over `train_range`: features, binning, scaling, the training window. Bin edges are a fit artifact, refit per fold. A scorer returns a per-row series (pnl and flips, or prediction and truth) and declares `metrics` and a direction per metric. The same `metrics` runs per fold and over the concatenated folds, so the aggregate needs no second definition.

Declarations are frozen dataclasses referencing registered step and scorer functions. An object is hashed by its canonical serialization: formatting changes nothing, a value change changes the id, a pipeline's name is a label outside the hash. No YAML, no closures. The step library's file sha is in every fit row.

## Objects

| object | identity | fields |
|---|---|---|
| dataset | hash(process, params, instrument, window, filters, targets) | rows, blob |
| pipeline | hash(fit, predict, config) | name, declaration, pickle |
| evaluation | hash(all fields) | declaration, pickle |
| fit | hash(dataset, pipeline, train_range) | cutoff, code sha, env lock, host fingerprint, duration, model blob |
| predictions | hash(fit, range) | fold, age, blob |
| score | hash(pipeline, evaluation) | one vector per (fold, age); one aggregate per age |
| decision | integer, append-only | kind (promote, reject), why, pipeline, evaluation, against: the baseline at the time with verdict and relative deltas |

Nothing is updated in place. The baseline is the latest promote decision under the evaluation. A pipeline's status (scored, baseline, superseded, rejected) is derived from the decisions. A comparison is a function of two aggregates, computed on read; the decision row keeps the one it acted on.

Invariants: a baseline moves only through a promote decision. A decision needs a score under the same evaluation. No fit's train_range ends after its fold's eval start minus the embargo. A scorer or schedule change is a new evaluation, so old scores stay readable and never compare across. Nothing is deleted.

## Workflow

Freeze a dataset. Declare pipelines and one evaluation in a Python module. `run` fits, predicts and scores every pipeline, memoized. `decide promote` seats the incumbent. `board` shows every scored pipeline with its verdict and per-metric deltas against the baseline. A person or an agent reads it and records `promote` or `reject` with a reason. `history` is the chain of promotions; `board <pipeline>` is one pipeline in full.

```
DS=$(fy freeze optiver.capture:freeze 20)
C="optiver.declarations --dataset $DS"
fy run $C
fy decide $C ridge_3m --kind promote --why "incumbent"
fy board $C
fy decide $C bonsai_lw --kind promote --why "pnl +7.6%, drawdown accepted"
fy history $C
```

Per project, three modules: how to read the raw data, the steps (features, models, scorers), the declarations (pipelines and the evaluation). Per idea, one more `Pipeline` in the declarations and `fy run` again.

## Storage

One directory: `forestry.sqlite` (seven tables, JSON bodies, insert only) and `blobs/sha256/` (datasets, models, predictions). Rows are few, so filtering happens in Python.

## Later

Each item was removed from draft 3 and returns only with a reason written here first.

| item | reopener |
|---|---|
| ordering rule for incomparable sets (priority order with a band) | when reading the board's deltas by hand costs more than the rule hides; the band made dominance intransitive, so any rule needs a seating story |
| staged scoring with gates (fit metrics, quick sim, full sim) | when a full simulation is expensive enough that screening must be recorded, not just done; until then two evaluations do it |
| execution config grids | when one fit must be scored under several thresholds in one evaluation; today a threshold is scorer config, so a different threshold is a different evaluation |
| tune step, heads | when a refresh tunes knobs per fold or a grid over readouts is the bottleneck; today a different n_iters is a different pipeline |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation whose schedule covers it |
| deploy, bundle, refresh seating | when a candidate goes to production from the ledger |
| remote executors (ssh, RunPod), agent journal, OpenLineage events | when a run leaves the laptop |
| capture layer under the dataset, candidate, comparison and baseline tables | when raw bytes must be shared across filter variants, or when a derived status is too slow to compute on read |
