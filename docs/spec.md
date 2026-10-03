# forestry: design spec (draft 4, 2026-10-03)

A local-first ledger for experimentation on frozen, time-ordered datasets. It records what was tried, under which evaluation, with what scores, and which recorded decision moved the baseline. People and agents write to the same ledger. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

Draft 4 trims draft 3 to record keeping. Staged funnels, gates, ordering rules, tune steps, heads, the sealed window, deployment bundles and campaign contracts were built, found to be too much surface, and removed; they are listed under "Later" with what would bring each back.

## Scope

In: dataset layers and hashes, pipeline and evaluation declarations, memoized fits and predictions, per-fold scores and aggregates, comparisons against a baseline, recorded decisions, read-back views, a shell command line. Out: automation that moves a baseline, a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## The pattern

An experiment is a triple on three independent, content-addressed axes. Nothing on one axis references another; the harness composes them.

| axis | holds | changes when |
|---|---|---|
| dataset | capture (process, instrument, window), filters, targets | new data or a labeling idea |
| pipeline | `fit` and `predict` steps and their config, including the training window | a modeling idea |
| evaluation | dataset id, schedule (cutoffs, eval windows, ages, embargo), scorer and its config, min_folds, compare_age | a validation or scoring idea |

Two memo points keep the cross product from materializing. A fit is keyed by (dataset, pipeline, train_range). Predictions are keyed by (fit, eval_range). Scores derive from predictions, the session's columns over the eval range and the scorer's config, so a scoring change refits nothing.

A candidate is (dataset, pipeline, evaluation). Comparability is "same evaluation id"; the harness refuses anything else.

## Contracts

```
fit(session, train_range, config) -> model
predict(model, session, range) -> predictions
score(predictions, session, range, config) -> ScoreResult(metrics, series)
```

`session` is the frozen dataset's raw columns, resident in memory; it is not any library's object. Anything that learns from data lives inside `fit` over `train_range`: features, binning, scaling, the training window. Bin edges are a fit artifact, refit per fold. A scorer declares a direction per metric and may declare `from_series`, which recomputes the aggregate over the concatenated fold series; otherwise the aggregate is the mean over folds.

Declarations are frozen dataclasses referencing registered step and scorer functions. An object is hashed by its canonical serialization: formatting changes nothing, a value change changes the id, a pipeline's name is a label outside the hash. No YAML, no closures. The step library's file sha is in every fit row.

## Objects

| object | identity | fields |
|---|---|---|
| capture | hash(process, params, instrument, bytes) | rows, blob |
| dataset | hash(capture, filters, targets) | family = (process, instrument), rows, blob |
| pipeline | hash(fit, predict, config) | name, declaration, pickle |
| evaluation | hash(all fields) | declaration, pickle |
| fit | hash(dataset, pipeline, train_range) | cutoff, code sha, env lock, host fingerprint, duration, model blob |
| predictions | hash(fit, range) | fold, age, blob |
| candidate | hash(dataset, pipeline, evaluation) | status: scored, baseline, superseded, rejected; reason |
| score | hash(candidate, fold, age) | metric vector; fold null marks the aggregate |
| comparison | uuid | challenger, baseline, verdict (dominates, dominated, incomparable), relative deltas, both metric vectors, decision id once acted on |
| decision | integer, append-only | kind (promote, reject), why, candidate, comparison |
| baseline | per evaluation | candidate, since decision |

Invariants: a baseline moves only through a promote decision. Two candidates compare only under one evaluation id with at least min_folds folds each. No fit's train_range ends after its fold's eval start minus the embargo. A scorer or schedule change is a new evaluation, so old comparisons go stale rather than silently comparable. Nothing is deleted; a refutation is a comparison row.

## Workflow

Freeze a dataset. Declare pipelines and one evaluation in a Python module. `run` fits, predicts and scores every pipeline, memoized. `decide promote` seats the incumbent. `compare` writes a verdict and per-metric deltas for each scored candidate against the baseline and moves nothing. A person or an agent reads the comparisons and records `promote` or `reject` with a reason. `board`, `history` and `why` read it back.

```
DS=$(fy freeze optiver.capture:freeze 20)
C="optiver.declarations --dataset $DS"
fy run $C
fy decide $C ridge_3m --kind promote --why "incumbent"
fy compare $C
fy decide $C bonsai_lw --kind promote --why "pnl +7.6%, drawdown accepted"
fy history $C
```

Per project, three modules: how to read the raw data, the steps (features, models, scorers), the declarations (pipelines and the evaluation). Per idea, one more `Pipeline` in the declarations and `fy run` again.

## Storage

One directory: `forestry.sqlite` (one table per object kind, JSON bodies) and `blobs/sha256/` (frames, models, predictions). Rows are few, so filtering happens in Python.

## Later

Each item was removed from draft 3 and returns only with a reason written here first.

| item | reopener |
|---|---|
| ordering rule for incomparable sets (priority order with a band) | when reading `compare` deltas by hand costs more than the rule hides; the band made dominance intransitive, so any rule needs a seating story |
| staged scoring with gates (fit metrics, quick sim, full sim) | when a full simulation is expensive enough that screening must be recorded, not just done; until then two evaluations do it |
| execution config grids | when one fit must be scored under several thresholds in one evaluation; today a threshold is scorer config, so a different threshold is a different evaluation |
| tune step, heads | when a refresh tunes knobs per fold or a grid over readouts is the bottleneck; today a different n_iters is a different pipeline |
| sealed window with a pass rule | when a deploy decision needs a one-shot held-out month; today a second evaluation whose schedule covers it |
| deploy, bundle, refresh seating | when a candidate goes to production from the ledger |
| remote executors (ssh, RunPod), agent journal, OpenLineage events | when a run leaves the laptop |
