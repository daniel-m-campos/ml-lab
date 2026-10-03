# User stories

Each story: a context, the steps as `fy` commands, what the ledger holds afterwards, and when it is done. Candidates are named by pipeline name when unique, else by id prefix.

## S1: freeze a dataset

Context: a new month of order-book captures for one instrument and sampling process.

1. `DS=$(fy freeze project.capture:freeze <args>)` calls the project's freeze function: it writes a capture row (process, params, instrument, bytes hash, rows), applies the declared filters and targets, and writes a dataset row.
2. The same call with the same inputs returns the same id and writes nothing.

Ledger after: one capture, one dataset, two blobs.

Done when: the printed id is stable across reruns.

## S2: run the pipelines

Context: `project/declarations.py` lists the pipelines and declares the evaluation (schedule, scorer, scorer config, min_folds).

1. `fy run project.declarations --dataset $DS` expands the schedule to folds, fits once per (pipeline, cutoff), predicts once per (fit, eval window), scores every fold and age, and writes one aggregate per age.
2. Rerunning computes nothing. Adding a pipeline to the module and rerunning fits only the new one.

Ledger after: one evaluation, N pipelines, N candidates, fits, predictions, scores.

Done when: the second run prints `fits 0, predictions 0`.

## S3: seat the incumbent and compare

Context: one of the pipelines is the model in production.

1. `fy decide project.declarations --dataset $DS ridge_3m --kind promote --why "incumbent"` records the decision and sets the baseline.
2. `fy compare project.declarations --dataset $DS` writes one comparison row per scored candidate: verdict, relative delta per metric, both metric vectors. Nothing moves.
3. The person reads the table and records `fy decide ... bonsai_lw --kind promote --why "..."` or `--kind reject --why "..."`. A promote links the comparison, moves the baseline and marks the old one superseded.

Done when: `fy history` shows the chain of promotions with the verdict and deltas each one acted on.

## S4: a scoring idea

Context: the cost assumption in the simulator changes.

1. Edit the scorer config in `declarations.py`; the evaluation id changes.
2. `fy run` again: zero fits, every candidate rescored under the new evaluation.
3. `fy compare` refuses a candidate scored under the old evaluation against a baseline under the new one.

Done when: both evaluations' boards are readable and nothing is overwritten.

## S5: iteration 100

Context: three months in.

1. `fy board` lists every candidate with status, latest verdict, aggregate metrics and the recorded reason.
2. `fy history` prints the promotions in order.
3. `fy why <candidate>` prints its config against the baseline's, its per-fold scores, its comparisons and decisions.

Done when: a newcomer can say what was tried, why each one lost, and what moved the baseline, from the ledger alone.

## Later

Stories removed with draft 4, kept as one line each: a staged funnel with recorded gates (S6), a sealed held-out month scored once (S7), deploy with a bundle naming its justification (S8), refresh seating the deployed candidate on the next dataset (S9), an agent campaign under a written contract (S10). Reopeners are in `docs/spec.md`, section "Later".
