# forestry: design spec (draft 3, 2026-10-03)

A local-first ledger for the lifecycle of experimentation on frozen, time-ordered datasets: compose pipelines, judge them under a declared evaluation, move a baseline only through a recorded comparison, deploy, refresh. Agents and humans write to the same ledger. Companions: `docs/user-stories.md`, `reports/Forestry MLOps landscape survey.md`.

## Scope

In: dataset layers and hashes, pipelines, evaluations, memoized fits and predictions, economic scores (simulations), Pareto comparisons, human decisions, baselines, deployments, refreshes, remote execution on a GPU box or a pod, an agent journal. Out: a DAG runner, a dashboard, the prod compile, feature stores, drift monitoring, non-temporal splits.

## The pattern

An experiment is a triple on three independent, content-addressed axes. Nothing on one axis references another; a harness composes them.

| axis | holds | changes when |
|---|---|---|
| dataset | capture, filters, targets and horizons | new data or a labeling idea |
| pipeline | the fit/predict implementation and its config, including how much history to train on | a modeling idea |
| evaluation | schedule of cutoffs and eval windows, ages, embargo, sealed window, staged scorers and gates, metrics, rule | a validation or scoring idea |

Two memo points keep the cross product from materializing. A fit is keyed by (dataset, pipeline, train_range); the pipeline hash covers its binning and preprocessing steps, so two pipelines that bin differently never share a fit. A prediction set is keyed by (fit, eval_range, head). Scores derive from predictions plus a scorer's config, so a scoring change refits nothing, and schedules that share a cutoff share the fit. Model bytes are optional: predictions and the fit key are kept always; bytes only for baselines, deployments and pins, since a fit is reproducible from its key, code sha and env lock.

Comparability is "same dataset, same evaluation". The harness refuses anything else.

## Knobs

| knob kind | lives in | cost |
|---|---|---|
| changes training: depth, learning rate, features, binning, training window | pipeline config, or a tune step's space | one fit per setting, or inner fits |
| readout of a trained model: iteration count, horizon column | head, declared by the pipeline | zero extra fits |
| changes trading, not the model: thresholds, sizing | execution config on the candidate | zero extra fits, one sim per setting |

## Pipeline contract

```
fit(session, train_range, config) -> model
predict(model, session, range, head) -> predictions   # one column per target or horizon
```

A head is a cheap readout of one trained model: an iteration count for a boosted model, a horizon column. The pipeline declares the heads it emits; a grid over heads costs one fit. Anything that changes training is a different pipeline, not a head.

`session` is the frozen dataset's raw columns held resident by the executor (host or device arrays); it is not a bonsai `Dataset` or any library's object. Ranges are time-ordered row ranges. Anything that learns from data lives inside `fit` over `train_range`: feature construction, binning, scaling, selection, the training window. Bin edges are therefore a fit artifact, refit per fold by default so no score row informs them; a pipeline may cache a binned view keyed by (dataset, binning step, train_range) and bonsai's resident-bin reuse is one such step, chosen by the pipeline, never assumed by the harness. `fit` may use less of `train_range` than it is given. A `tune` step inside `fit` searches a space (hyperparameters, early-stopping patience, training window) over an inner time-ordered split of `train_range` with the embargo applied, then refits on the full `train_range` with the chosen setting; the setting per fold is a fit artifact, recorded like bin edges. Selection inside the process is scored by walk-forward as part of the process; selection outside it is a grid of pipelines, whose contamination the sealed window catches. Rule: a knob tuned per refresh belongs in a tune step; a knob learned once about the dataset is a grid. Pipelines, evaluations, stages and gates are typed Python objects (dataclasses) referencing registered step and scorer functions; there is no YAML. An object is hashed by its canonical serialization, so formatting changes nothing and a value change changes the id; a declaration may hold only values and references to registered steps, never closures, and the harness rejects what it cannot serialize. The step library's code sha is in every fit key. Grids are comprehensions over `with_config`. A second contract, `fit_sequence(session, schedule, config) -> models`, exists for online or cross-fold ideas; an evaluation lists the contracts it accepts. No third contract.

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
| stages | ordered scorers, each with a cost class and a gate; see below | fit metrics (human gate), quick sim grid (Pareto front, cap 5), full sim |
| metrics | per stage; the last stage's vector is business-oriented (pnl, sharpe, max drawdown, turnover) | |
| aggregate | over folds: concatenated series for pnl and sharpe, worst drawdown, equal weights | |
| rule | pareto on the aggregate; tie policy human | |
| contracts | fit/predict, fit_sequence | fit/predict |

Any field change is a new evaluation id. A scored sealed window becomes a public fold; time supplies the next seal.

### Stages

| stage | scorer | reads | gate |
|---|---|---|---|
| 1 | fit metrics from predictions: correlation with target, hit rate, per-horizon loss, setting stability across folds | predictions | human, or threshold and top-k once one is known |
| 2 | quick simulator over a grid of execution configs; one fit yields one candidate per grid point | predictions | Pareto front of the grid, capped at 5 |
| 3 | full simulator on the survivors | predictions | dominance against the baseline |

Gate kinds: threshold, top-k, Pareto front, human. A human gate writes a decision row. A candidate that stops at a stage keeps its score rows there and a `stopped_at` with the gate's reason, so screening is recorded, not just done. Comparisons against the baseline are valid only at the stage the baseline holds, normally the last.

## Objects

| object | identity | fields |
|---|---|---|
| capture | hash(process, instrument, window) | process params, instrument, window, bytes hash, rows |
| dataset | hash(capture, filters, targets) | capture id, filter rules, targets and horizons, bytes hash; family = (process, instrument) |
| pipeline | hash(step graph, config, library sha) | steps incl. tune, config, heads, contract |
| evaluation | hash(all fields above) | see table |
| fit | hash(dataset, pipeline, train_range) | code sha, env lock, host fingerprint, executor, cost, duration, artifacts (bin edges, chosen settings), bundle hash or null |
| predictions | hash(fit, eval_range, head) | columns, blob hash |
| candidate | hash(dataset, pipeline, head, exec config) | which prediction column the simulator trades, execution params, stopped_at |
| score | (candidate, evaluation, stage, fold, age) | metric vector; aggregate rows flagged |
| comparison | uuid | challenger, baseline, evaluation, verdict: dominates / dominated / incomparable, decision id |
| decision | integer, append-only | rationale, comparison or candidate ids, kind: gate / promote / reject / deploy / retire / seal-pass / seal-fail, prod feedback links |
| baseline | per (family, evaluation) | current candidate, since decision |
| deployment | uuid | bundle hash, env, deployed at, retired at, decision id |
| campaign | uuid | dataset, evaluation, contract (budget, stop rule, approvals, fail_outcome) |

Invariants: a baseline moves only through a comparison with verdict dominates or a promote decision. Two candidates compare only under one evaluation id, at the same stage, with at least min_folds folds each. No fit's train_range ends after its fold's eval start minus embargo. Nothing scores the sealed window before a seal decision. A simulator or schedule change is a new evaluation, so old comparisons go stale rather than silently comparable. Refutations are comparison rows, never deletions.

## Workflows

Campaign: freeze dataset, declare evaluation, run pipelines (harness expands the schedule, memoizes fits and predictions per head on the resident session, scores stage by stage, applies gates, writes scores), compare survivors against the baseline, decide, score the sealed window once, deploy or apply fail_outcome.

Deployment model: the evaluation names it, last fold's fit or a refit on the full window after promotion; that refit is a fit with no score.

Refresh: new capture from the same family with a later window; the deployed candidate is seated as baseline and scored on the new folds first; the previous sealed month is now public; the newest month is sealed. Prod scores attach to the deploy decision as feedback.

Comparison: dominance on the aggregate vector. Incomparable opens a human decision with a written rationale.

## Storage

One git repo: `forestry.sqlite`, `blobs/sha256/` (predictions, bundles, manifests, eval outputs), `decisions.md` rendered from the decision table. Run events emitted as OpenLineage RunEvent JSON with a `forestry_*` facet pinned to a schema sha. Bundle: `model.msgpack` plus `bundle.json` (dataset, pipeline, fit key, exec config, evaluation, metric vector, code sha, env lock, host fingerprint).

## Execution

One executor interface: submit(work, dataset ref) -> handle with poll, logs, pull, teardown. Instances: local, ssh (the GPU box), runpod (REST v2 stock ladder, idle watchdog that terminates, cost and duration into the fit row). Host fingerprint on every fit.

## Agents

An agent uses the CLI and Python API. The journal is a node per attempt: parent, plan, diff, fit ids, outcome, is_buggy, verdict. The harness owns ranges, the clock, the scorers, the gates and the sealed window; the agent writes `fit` and `predict`. The stages are the agent's budget: many stage-1 attempts, few stage-2 grids, stage 3 behind a human gate when the contract says so. A claim without a score row counts for nothing. The campaign contract is written by the human first.

## Open

1. Objective count before the incomparable set swamps the human; a cap or priority order per evaluation.
2. Whether bonsai's bench harness becomes the first local executor or stays a consumer.
3. Minimum slice: dataset, pipeline, evaluation, fit, predictions, candidate, score, comparison, baseline, decision on SQLite with the local executor; remote executors and the journal second.
