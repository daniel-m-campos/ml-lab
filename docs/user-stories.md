# User stories

Each story: a context, the steps as `fy` commands, what the ledger holds afterwards, and when it is done. Pipelines are named by name when unique, else by id prefix.

## S1: freeze a dataset

Context: a new month of order-book captures for one instrument and sampling process.

1. `DS=$(fy freeze project.capture:freeze <args>)` calls the project's freeze function: it loads the rows, applies the declared filters, checks the targets and writes one dataset row (process, params, instrument, window, filters, targets, rows, blob).
2. The same call with the same inputs returns the same id and writes nothing.

Ledger after: one dataset, one blob.

Done when: the printed id is stable across reruns.

## S2: run the pipelines

Context: `project/declarations.py` lists the pipelines and declares the evaluation (schedule, scorer, scorer config, min_folds).

1. `fy run project.declarations --dataset $DS` expands the schedule to folds, fits once per (pipeline, cutoff), predicts once per (fit, eval window), and writes one score row per pipeline holding every fold's vector and one aggregate per age.
2. Rerunning computes nothing. Adding a pipeline to the module and rerunning fits only the new one.

Ledger after: one evaluation, N pipelines, N score rows, fits, predictions.

Done when: the second run prints `fits 0, predictions 0`.

## S3: seat the incumbent and decide

Context: one of the pipelines is the model in production.

1. `fy decide project.declarations --dataset $DS ridge_3m --kind promote --why "incumbent"` records the first promotion; the baseline is now whatever was promoted last.
2. `fy board project.declarations --dataset $DS` shows every scored pipeline with its Pareto verdict and relative delta per metric against the baseline. Reading it writes nothing.
3. The person records `fy decide ... bonsai_lw --kind promote --why "..."` or `--kind reject --why "..."`. The decision row keeps the verdict and deltas it was made against.

Done when: `fy history` shows the chain of promotions, each with what it was made against.

## S4: a scoring idea

Context: the cost assumption in the simulator changes.

1. Edit the scorer config in `declarations.py`; the evaluation id changes.
2. `fy run` again: zero fits, every candidate rescored under the new evaluation.
3. `fy decide` refuses a pipeline that has no score under the new evaluation.

Done when: both evaluations' boards are readable and nothing is overwritten.

## S5: iteration 100

Context: three months in.

1. `fy board` lists every scored pipeline with status, verdict and deltas against the baseline, aggregate metrics and the last recorded reason.
2. `fy history` prints the promotions in order.
3. `fy board <pipeline>` prints one pipeline: config against the baseline's, per-fold scores, every decision on it.

Done when: a newcomer can say what was tried, why each one lost, and what moved the baseline, from the ledger alone.

## Later

Stories removed with draft 4, kept as one line each: a staged funnel with recorded gates (S6), a sealed held-out month scored once (S7), deploy with a bundle naming its justification (S8), refresh seating the deployed candidate on the next dataset (S9), an agent campaign under a written contract (S10). Reopeners are in `docs/spec.md`, section "Later".
