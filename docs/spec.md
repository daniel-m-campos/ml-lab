# forestry: design spec (draft 1, 2026-10-03)

A local-first ledger for the lifecycle of experimentation on frozen datasets: compose pipelines, compare candidates under a declared protocol, move a baseline only through a recorded comparison, deploy, refresh. Agents and humans write to the same ledger. Companion: `reports/Forestry MLOps landscape survey.md`.

## Scope

In: dataset recipes and hashes, pipeline composition and step caching, trials, economic evals, Pareto comparisons, human decisions, baselines, deployments, refreshes, remote execution on a GPU box or a pod, an agent journal. Out: a DAG runner (DVC or plain Python drives steps), a dashboard, the prod compile, feature stores, drift monitoring.

## Objects

| object | identity | fields |
|---|---|---|
| capture | hash(process, instrument, window) | process name and params, instrument, window [t0, t1), bytes hash, size, rows |
| dataset | hash(capture, filters, target) | capture id, filter recipe, target definition and horizons, bytes hash; family = (process, instrument) |
| pipeline | hash(step graph) | ordered steps, each: name, code sha, config hash, inputs |
| protocol | hash(dataset id, window, simulator, sim config, metric names, rule) | eval dataset, eval window, simulator version and config, metric vector, dominance rule (pareto), tie policy (human) |
| trial | uuid | dataset id, pipeline id, seeds, cutoff, code sha, env lock hash, host fingerprint, executor, cost, duration, status, parent trial |
| candidate | hash(model bundle, execution config) | trial id, bundle hash, execution config hash |
| eval | (candidate, protocol) | metric vector, per-fold or per-window values, run id |
| comparison | uuid | challenger candidate, baseline candidate, protocol id, verdict: dominates / dominated / incomparable, decision id if human |
| decision | integer, append-only | text rationale, comparison ids, kind: promote / reject / deploy / retire, prod feedback links |
| baseline | per (family, protocol) | current candidate, since decision |
| deployment | uuid | bundle hash, target env, deployed at, retired at, decision id |

Invariants: a baseline moves only through a comparison whose verdict is dominates or whose decision is promote. Two candidates compare only under one protocol id. A trial's inputs carry no rows past its cutoff. A changed simulator yields a new protocol id, so old comparisons go stale instead of silently comparable. Refutations are recorded comparisons, not deletions.

## Dataset layers

capture (sampling process, instrument, window) -> filtered (outliers, broken captures, blackouts) -> labeled (target, horizons). Each layer hashes its recipe and bytes. Features are pipeline steps, never dataset identity. Parquet is hashed as stored bytes once at freeze.

## Pipelines and caching

A step is a pure function of (input hashes, config hash, code sha). Cache key is that triple. Cache value is bytes on disk or "resident in this device session". A session holds a dataset on device across steps (CV, selection, fit) and records each step boundary without forcing a host round trip. The economic simulation is a step like any other.

## Workflows

Campaign: freeze a dataset, declare a protocol, run trials, eval candidates, compare against the baseline, decide. Ends in a deploy or reject decision.

Refresh: new capture from the same family with a later window, new dataset id, same or new protocol. The deployed candidate is seated as baseline. Same campaign otherwise.

Comparison: compute the metric vector for both candidates under the protocol. Pareto dominance resolves automatically. Incomparable opens a decision for a human; its rationale is text; prod feedback attaches to it later so the rule can formalize from evidence.

## Storage

One directory in a git repo: `forestry.sqlite` (objects above), `blobs/sha256/` (bundles, manifests, eval outputs), `decisions.md` (rendered from the decision table). Run events are emitted as OpenLineage RunEvent JSON with a `forestry_*` facet pinned to a schema sha. The bundle is `model.msgpack` plus `bundle.json` (dataset id, pipeline id, feature list, execution config, protocol id, metric vector, code sha, env lock, host fingerprint).

## Execution

One executor interface: submit(step graph, dataset ref) -> job handle with poll, logs, pull, teardown. Instances: local, ssh (the GPU box), runpod (REST v2 with stock ladder, idle watchdog that terminates, cost and duration into the trial). Host fingerprint on every trial: CPU model, cores, caches, RAM, GPU and driver, cgroup cpu.max and cpuset, container image digest.

## Agents

An agent is a user of the CLI and Python API. The journal is the trial table plus a node record: parent, plan, diff, outcome, is_buggy, verdict. The eval split and simulator run harness-side, outside the agent's writable tree. A trial without a run id counts for nothing. A campaign carries a contract file the human writes first: protocol, budget, stop rule, what needs approval.

## Not now

Generic non-temporal splits. Multi-user server. Dashboard. Feature store. Drift monitoring. Prod compile. Cross-protocol comparison.

## Open

1. Pareto with how many objectives before the incomparable set swamps the human; a cap or a priority order per protocol.
2. Trial granularity for CV: one trial per fold set or one per seed.
3. Whether bonsai's bench harness becomes forestry's first executor or stays a consumer.
4. Minimum viable slice: capture, dataset, pipeline, trial, candidate, eval, comparison, baseline, decision on SQLite with the local executor; remote executors and the agent journal second.
