# User stories

Each story: a context, the steps as `fy` commands, what the ledger holds afterwards, and when it is done. `C="project.declarations --dataset $DS"` throughout. Pipelines are named by name when unique, else by id prefix.

## S1: open the dataset account

Context: a new month of order-book captures for one instrument and sampling process.

1. `DS=$(fy freeze project.capture:freeze <args>)` loads the rows, applies the declared filter steps, checks the targets, stores the bytes under their sha and writes one dataset row with the recipe.
2. The same call with the same inputs returns the same id and writes nothing.

Ledger after: one dataset row, one blob.

Done when: the printed id is stable across reruns.

## S2: the first run opens the evaluation and posts entries

Context: `project/declarations.py` lists the pipelines and declares the evaluation.

1. `fy run $C` writes the evaluation row, its metrics with directions, and the schedule as fold and eval_window rows; then a run row with the git commit, dirty flag, environment lock and host; then, per pipeline, fits per cutoff, predictions per eval window, and one entry with fold_score and aggregate_score rows.
2. `fy run $C` again posts a new run whose entries reuse every fit and prediction. Adding a pipeline to the module fits only the new one.

Ledger after: one evaluation, N pipelines, two runs, 2N entries, fits, predictions.

Done when: the second run prints zero fits and zero predictions.

## S3: seat the incumbent and decide

Context: one of the pipelines is the model in production.

1. `fy decide $C ridge_3m --kind promote --why "incumbent"` writes a decision on that pipeline's latest entry with nothing to compare against. The baseline view resolves to it.
2. `fy board $C` reads the comparison view: each pipeline's latest entry against the baseline, Pareto verdict, relative delta per metric. Reading writes nothing.
3. `fy decide $C bonsai_lw --kind promote --why "..."` or `--kind reject --why "..."` writes a decision carrying the baseline entry, verdict and deltas it saw.

Done when: `fy history $C` shows the promotes in order, each with what it was made against, and the old baseline reads as superseded.

## S4: a scoring idea

Context: the cost assumption in the simulator changes.

1. Edit the scorer config in `declarations.py`; the evaluation id changes.
2. `fy run $C`: a new evaluation account, zero fits, every pipeline rescored from its memoized predictions.
3. `fy decide` on the new evaluation starts its own baseline; entries under the old evaluation are untouched and never compare across.

Done when: both boards read and nothing was overwritten.

## S5: code evolves

Context: a feature in `steps.py` changes.

1. Commit, `fy run $C`. Git reports which step files moved since each fit's commit; the pipelines that use them refit at HEAD as new fit rows and post new entries in the new run. The rest reuse.
2. `fy board $C` shows the new entries. Decisions made on the old entries stand, pointing at their run and commit.

Done when: the board's deltas reflect the new code and `fy board $C <pipeline>` shows both entries' runs.

## S6: iteration 100

Context: three months in.

1. `fy board $C` lists every pipeline's latest entry with status, verdict, deltas, aggregate metrics and the last recorded reason.
2. `fy history $C` prints the promotes in order.
3. `fy board $C <pipeline>` prints one pipeline: config against the baseline's, per-fold scores, every entry with its commit, every decision on it.
4. To reproduce a score: its run names the commit and environment, its fit names the dataset blob and train range. Check out, load, rerun the declaration.

Done when: a newcomer can say what was tried, why each one lost, what moved the baseline, and can rebuild any number, from the ledger and git alone.
