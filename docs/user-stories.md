# User stories

Template. One story per heading. Keep each under fifteen lines.

```
## S<n>. <who> <does what> so that <why>
Context: one or two lines of situation.
Steps: numbered, each naming the command or API call and the object it writes.
Ledger after: which rows exist that did not before.
Done when: the observable fact that closes the story.
```

Commands below are the proposed CLI; names are provisional. `fy` is the binary.

## S1. Researcher freezes a dataset so that every later result points at one set of bytes

Context: a new capture of ES futures, 2026-07 to 2026-09, from the standard sampler, with the usual blackout and broken-capture filters, labeled with a 30 s and a 120 s horizon.

Steps:
1. `fy capture freeze --process sampler-v3 --instrument ES --window 2026-07-01:2026-10-01 es-q3.parquet` writes a capture row with the bytes hash.
2. `fy dataset freeze --capture <id> --filters filters/std.yaml --target targets/ret-30s-120s.yaml` writes a dataset row; the filtered and labeled bytes land in `blobs/`.
3. `git commit` the two manifests.

Ledger after: one capture, one dataset, both with recipe and bytes hashes.

Done when: `fy dataset show <id>` prints the recipe, hashes and row count, and a rerun of step 2 with the same inputs returns the same id and writes nothing.

## S2. Researcher declares how candidates will be judged so that no comparison is argued after the fact

Context: the campaign on S1's dataset will be judged by the taker simulator on the last month of the window.

Steps:
1. `fy protocol declare --dataset <id> --eval-window 2026-09-01:2026-10-01 --sim taker-sim@1.4 --sim-config sim/es-taker.yaml --metrics pnl,sharpe,max_dd,turnover --rule pareto --tie human` writes a protocol row.

Ledger after: one protocol with a hash over every input.

Done when: changing any input, including the simulator version, yields a different protocol id.

## S3. Researcher runs a modeling idea end to end so that the result is comparable to everything else on the dataset

Context: a new order-book imbalance feature set, depthwise bonsai, five seeds, data resident on the GPU box.

Steps:
1. Edit `recipes/imbalance-v2.yaml`: steps feature-imbalance -> select-topk -> bonsai-depthwise, train window 1y. The recipe implements fit/predict and knows nothing about folds.
2. `fy run --dataset <id> --recipe recipes/imbalance-v2.yaml --protocol <id> --seeds 5 --executor gpubox`: the harness expands the protocol's schedule, writes one fit row per cutoff (memoized by dataset, recipe, train_range) and one predictions row per eval window and age, all on the resident session.
3. `fy candidate add --dataset <id> --recipe <id> --exec-config exec/es-taker-a.yaml` names the prediction column to trade; `fy eval --candidate <id> --protocol <id>` runs the simulator harness-side and writes per-fold and aggregate eval rows.

Ledger after: N fits, N predictions, one candidate, N+1 evals.

Done when: `fy evals --dataset <id> --protocol <id>` lists the candidate beside every earlier one, and rerunning step 2 unchanged computes no fit.

## S4. Researcher compares the new candidate to the baseline so that the baseline moves only on evidence

Context: S3's candidate against the current baseline for (ES, protocol from S2).

Steps:
1. `fy compare --challenger <cand> --baseline current --protocol <id>` computes dominance over the metric vector and writes a comparison row.
2. If `dominates`: the baseline row advances and a decision row of kind promote is written with the comparison id.
3. If `incomparable` (higher pnl, higher drawdown): `fy decide <comparison> --kind promote|reject --why "..."` writes the decision with the rationale; the baseline moves only on promote.
4. If `dominated`: nothing moves; the comparison stays as the record that the idea was tried.

Ledger after: one comparison, zero or one decision, baseline possibly advanced.

Done when: `fy baseline --family ES --protocol <id>` names the candidate and the decision that seated it.

## S5. Researcher asks what has been tried so that no idea is rerun or lost

Context: three months into the campaign, 140 fits.

Steps:
1. `fy fits --dataset <id> --sort sharpe` lists fits with recipe name, seeds, metric vector, verdict against the baseline at the time.
2. `fy history --baseline ES/<protocol>` prints the chain of promotions with their decisions and rationales.
3. `fy why <fit>` prints the recipe, step configs and diffs against the baseline's fit.

Ledger after: unchanged.

Done when: the answer to "did we try X" is one query, and every rejected idea has a comparison row saying why.

## S6. Researcher deploys the baseline so that prod and the ledger agree on what is running

Context: the campaign ends; the baseline candidate goes to prod.

Steps:
1. `fy decide --kind deploy --candidate <id> --why "..."` writes the decision.
2. `fy deploy --candidate <id> --env prod-es` writes a deployment row with the bundle hash and time; the proprietary compile step reads the bundle from `blobs/` and is out of scope.

Ledger after: one decision, one deployment.

Done when: `fy deployments --env prod-es` shows the bundle hash, and the bundle's `bundle.json` names the dataset, recipe, protocol and metrics that justified it.

## S7. Researcher refreshes a deployed model so that the incumbent is the bar to clear

Context: a quarter has passed; a new capture of ES from the same sampler.

Steps:
1. S1 again with the new window: new capture and dataset ids, same family (sampler-v3, ES).
2. `fy protocol declare ...` on the new dataset; `fy baseline seat --family ES --protocol <new> --candidate <deployed>` seats the deployed candidate as baseline.
3. `fy eval --candidate <deployed> --protocol <new>` scores the incumbent on the new window.
4. S3 and S4 for each challenger. A promote decision leads to S6.

Ledger after: new dataset, protocol, baseline seat, evals, comparisons.

Done when: the incumbent's eval on the new window exists before any challenger's, and the deployment row for the old bundle gains a retired-at time when a new one goes out.

## S8. Agent runs a night of experiments so that the human reviews evidence, not transcripts

Context: a contract file sets the protocol, a budget of 12 GPU-hours, a stop rule of 40 fits or three consecutive dominated comparisons, and "promotion requires a human".

Steps:
1. `fy campaign open --dataset <id> --protocol <id> --contract contracts/night-1.yaml` writes the campaign and its contract.
2. The agent loops S3 and S4 through the same CLI; each fit row carries the journal node (parent, plan, diff, outcome, is_buggy). Evals run harness-side; the eval window is outside the agent's writable tree.
3. The run stops on the contract's rule. `fy campaign report <id>` prints fits, comparisons, cost and the incomparable set awaiting decisions.
4. The human works through S4 step 3 on the incomparable set.

Ledger after: one campaign, N fits with journal nodes, N comparisons, zero promotions.

Done when: every claim in the agent's report resolves to a fit id and an eval row, and nothing moved the baseline without a human decision.

## S9. Researcher changes the simulator so that stale comparisons cannot pass as current

Context: taker-sim 1.5 corrects fee handling. No refit happens: evals derive from cached predictions.

Steps:
1. `fy protocol declare ... --sim taker-sim@1.5` yields a new protocol id.
2. `fy eval` re-scores the baseline and any candidates still of interest under the new protocol.
3. Old comparisons remain in the ledger under the old protocol id and are not consulted for the new baseline.

Ledger after: new protocol, new evals, new baseline seat.

Done when: `fy compare` refuses a challenger evaluated under 1.4 against a baseline evaluated under 1.5.
