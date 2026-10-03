# forestry: design spec (draft 2, 2026-10-03)

A local-first ledger for the lifecycle of experimentation on frozen, time-ordered datasets: compose pipelines, judge them under a declared evaluation, move a baseline only through a recorded comparison, deploy, refresh. Agents and humans write to the same ledger. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

## Scope

In: dataset layers and hashes, pipelines, evaluations, memoized fits and predictions, economic scores (simulations), Pareto comparisons, human decisions, baselines, deployments, refreshes, remote execution on a GPU box or a pod, an agent journal. Out: a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## The pattern

An experiment is a triple on three independent, content-addressed axes. Nothing on one axis references another; a harness composes them.

| axis | holds | changes when |
|---|---|---|
| dataset | capture, filters, targets and horizons | new data or a labeling idea |
| pipeline | the fit/predict implementation and its config, including how much history to train on | a modeling idea |
| evaluation | schedule of cutoffs and eval windows, ages, embargo, sealed window, simulator and config, metrics, rule | a validation or eval idea |

Two memo points keep the cross product from materializing. A fit is keyed by (dataset, pipeline, train_range); the pipeline hash covers its binning and preprocessing steps, so two pipelines that bin differently never share a fit. A prediction set is keyed by (fit, eval_range). Evals derive from predictions plus simulator config, so an eval change refits nothing, and schedules that share a cutoff share the fit. Model bytes are optional: predictions and the fit key are kept always; bytes only for baselines, deployments and pins, since a fit is reproducible from its key, code sha and env lock.

Comparability is "same dataset, same evaluation". The harness refuses anything else.

## Pipeline contract

```
fit(session, train_range, config) -> model
predict(model, session, range) -> predictions   # one column per target or horizon
```

`session` is the frozen dataset's raw columns held resident by the executor (host or device arrays); it is not a bonsai `Dataset` or any library's object. Ranges are time-ordered row ranges. Anything that learns from data lives inside `fit` over `train_range`: feature construction, binning, scaling, selection, the training window. Bin edges are therefore a fit artifact, refit per fold by default so no score row informs them; a pipeline may cache a binned view keyed by (dataset, binning step, train_range) and bonsai's resident-bin reuse is one such step, chosen by the pipeline, never assumed by the harness. `fit` may use less of `train_range` than it is given. Pipelines are configs over a versioned step library; the library's code sha is in every fit key. A second contract, `fit_sequence(session, schedule, config) -> models`, exists for online or cross-fold ideas; an evaluation lists the contracts it accepts. No third contract.

## Evaluation

| field | meaning | default |
|---|---|---|
| schedule | cutoffs and eval windows, by calendar date | monthly cutoffs, one-month eval windows |
| ages | eval windows at increasing distance from each cutoff, to measure decay | 1, 2, 3 months |
| embargo | gap between train end and eval start | longest horizon; practice is day splits, so zero overlap |
| min_folds | folds required before a comparison is valid | 3 |
| sealed | final window scored once at decision time | last month of the dataset |
| pass_rule | sealed-window metrics must fall inside the candidate's per-fold band | empirical fold range |
| fail_outcome | chosen before unsealing: no deploy, deploy incumbent, or re-seal next month | no deploy |
| simulator | name, version, config hash | |
| metrics | vector, business-oriented (pnl, sharpe, max drawdown, turnover) | |
| aggregate | over folds: concatenated series for pnl and sharpe, worst drawdown, equal weights | |
| rule | pareto on the aggregate; tie policy human | |
| contracts | fit/predict, fit_sequence | fit/predict |

Any field change is a new evaluation id. A scored sealed window becomes a public fold; time supplies the next seal.

## Objects

| object | identity | fields |
|---|---|---|
| capture | hash(process, instrument, window) | process params, instrument, window, bytes hash, rows |
| dataset | hash(capture, filters, targets) | capture id, filter rules, targets and horizons, bytes hash; family = (process, instrument) |
| pipeline | hash(step graph, config, library sha) | steps, config, contract |
| evaluation | hash(all fields above) | see table |
| fit | hash(dataset, pipeline, train_range) | code sha, env lock, host fingerprint, executor, cost, duration, bundle hash or null |
| predictions | hash(fit, eval_range) | columns, blob hash |
| candidate | hash(dataset, pipeline, exec config) | which prediction column the simulator trades, execution params |
| score | (candidate, evaluation, fold, age) | metric vector; aggregate rows flagged |
| comparison | uuid | challenger, baseline, evaluation, verdict: dominates / dominated / incomparable, decision id |
| decision | integer, append-only | rationale, comparison ids, kind: promote / reject / deploy / retire / seal-pass / seal-fail, prod feedback links |
| baseline | per (family, evaluation) | current candidate, since decision |
| deployment | uuid | bundle hash, env, deployed at, retired at, decision id |
| campaign | uuid | dataset, evaluation, contract (budget, stop rule, approvals, fail_outcome) |

Invariants: a baseline moves only through a comparison with verdict dominates or a promote decision. Two candidates compare only under one evaluation id and with at least min_folds folds each. No fit's train_range ends after its fold's eval start minus embargo. Nothing scores the sealed window before a seal decision. A simulator or schedule change is a new evaluation, so old comparisons go stale rather than silently comparable. Refutations are comparison rows, never deletions.

## Workflows

Campaign: freeze dataset, declare evaluation, run pipelines (harness expands the schedule, memoizes fits and predictions on the resident session, runs the simulator, writes scores), compare against the baseline, decide, score the sealed window once, deploy or apply fail_outcome.

Deployment model: the evaluation names it, last fold's fit or a refit on the full window after promotion; that refit is a fit with no score.

Refresh: new capture from the same family with a later window; the deployed candidate is seated as baseline and scored on the new folds first; the previous sealed month is now public; the newest month is sealed. Prod scores attach to the deploy decision as feedback.

Comparison: dominance on the aggregate vector. Incomparable opens a human decision with a written rationale.

## Storage

One git repo: `forestry.sqlite`, `blobs/sha256/` (predictions, bundles, manifests, eval outputs), `decisions.md` rendered from the decision table. Run events emitted as OpenLineage RunEvent JSON with a `forestry_*` facet pinned to a schema sha. Bundle: `model.msgpack` plus `bundle.json` (dataset, pipeline, fit key, exec config, evaluation, metric vector, code sha, env lock, host fingerprint).

## Execution

One executor interface: submit(work, dataset ref) -> handle with poll, logs, pull, teardown. Instances: local, ssh (the GPU box), runpod (REST v2 stock ladder, idle watchdog that terminates, cost and duration into the fit row). Host fingerprint on every fit.

## Agents

An agent uses the CLI and Python API. The journal is a node per attempt: parent, plan, diff, fit ids, outcome, is_buggy, verdict. The harness owns ranges, the clock, the simulator and the sealed window; the agent writes `fit` and `predict`. A claim without an score row counts for nothing. The campaign contract is written by the human first.

## Open

1. Objective count before the incomparable set swamps the human; a cap or priority order per evaluation.
2. Whether bonsai's bench harness becomes the first local executor or stays a consumer.
3. Minimum slice: dataset, pipeline, evaluation, fit, predictions, candidate, eval, comparison, baseline, decision on SQLite with the local executor; remote executors and the journal second.
