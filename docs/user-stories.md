# User stories

Each story: a context, the steps as `fy` commands, what the log holds afterwards, and when it is done. `C="project.declarations --dataset $DS"` throughout. Pipelines are named by name when unique, else by id prefix.

## S1: open the dataset account

Context: a new month of order-book captures for one instrument and sampling process.

1. `DS=$(fy freeze project.capture:freeze <args>)` loads the rows, applies the declared filter steps, checks the targets, stores the bytes under their sha and appends one `dataset_frozen` event with the recipe.
2. The same call with the same inputs returns the same id and writes nothing.

Log after: one event, one blob.

Done when: the printed id is stable across reruns.

## S2: the first run opens the evaluation and posts entries

Context: `project/declarations.py` lists the pipelines and declares the evaluation.

1. `fy run $C` appends `evaluation_declared` with the metric directions and the schedule expanded into folds and windows; then `run_started` with the commit, dirty flag, environment lock and host; then, per pipeline, `fit_computed` per cutoff carrying the import shas, `predictions_computed` per eval window, and one `entry_scored` keyed by the prediction ids.
2. `fy run $C` again appends one `run_started` and nothing else: every fit and prediction is reused and every entry already exists. Adding a pipeline to the module fits only the new one.

Log after: one evaluation, N pipelines, two runs, N entries, fits, predictions.

Done when: the second run prints zero fits and zero predictions.

## S3: seat the incumbent and decide

Context: one of the pipelines is the model in production.

1. `fy decide $C ridge_3m --kind promote --why "incumbent"` appends a decision on that pipeline's entry with nothing to compare against. The baseline view resolves to it.
2. `fy board $C` reads the comparison view: each pipeline's latest entry against the baseline, Pareto verdict, relative delta per metric. Reading writes nothing.
3. `fy decide $C bonsai_lw --kind promote --why "..."` or `--kind reject --why "..."` appends a decision carrying the baseline entry, verdict and deltas it saw.

Done when: `fy history $C` shows the promotes in order, each with what it was made against, and the old baseline reads as superseded.

## S4: a scoring idea

Context: the cost assumption in the simulator changes.

1. Edit the scorer config in `declarations.py`; the evaluation id changes.
2. `fy run $C`: a new evaluation account, zero fits, every pipeline rescored from its memoized predictions.
3. `fy decide` on the new evaluation starts its own baseline; entries under the old evaluation are untouched and never compare across.

Done when: both boards read and nothing was overwritten.

## S5: code evolves

Context: a feature in `steps.py` changes.

1. `fy run $C`, committed or not. Each fit recorded the git blob sha of every module it imported; the ones whose shas no longer match refit, their predictions change, and each affected pipeline gets a new entry with no decisions. A dirty tree is recorded with its diff. The rest reuse.
2. `fy board $C` shows the new entries as scored, including the former baseline's pipeline: its approval was pinned to the old content. Decisions on the old entries stand, pointing at their run, commit and shas.

Done when: the board asks for review again and `fy board $C <pipeline>` shows both entries' runs.

## S6: iteration 100

Context: three months in.

1. `fy board $C` lists every pipeline's latest entry with status, verdict, deltas, aggregate metrics and the last recorded reason.
2. `fy history $C` prints the promotes in order.
3. `fy board $C <pipeline>` prints one pipeline: config against the baseline's, per-fold scores, every entry with its commit, every decision on it.
4. To reproduce a score: the entry names its run and predictions, the run its commit and lock, each fit its import shas and model sha. Check out, sync the lock, load the dataset blob, rerun, compare shas.
5. For analytics, attach the log in DuckDB and query the same views, or load prediction blobs by sha for per-row tests.

Done when: a newcomer can say what was tried, why each one lost, what moved the baseline, and can rebuild any number, from the log and git alone.
