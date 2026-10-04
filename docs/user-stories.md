# User stories

Each story: a context, the steps, what the log holds afterwards, and when it is done. `D=project/experiment.py` throughout; `sql` is `sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite`. S3 and S6 describe decisions, which are designed but not built; see Later in the spec.

## S1: record the dataset

Context: a new month of order-book captures for one instrument and sampling process.

1. `lab ingest project/dataset.py <args>` calls the module's `dataset` function, which loads the rows, applies the declared filter steps, checks the targets, stores the bytes under their sha and appends one `dataset_recorded` event with the recipe.
2. The same call with the same inputs returns the same id and writes nothing.

Log after: one event, one blob.

Done when: the printed id is stable across reruns.

## S2: the first run declares the evaluation and posts scores

Context: `project/experiment.py` lists the pipelines and declares the evaluation.

1. `lab run $D` appends `evaluation_declared` with the metric directions and the schedule expanded into folds and windows; then `run_started` with the commit, dirty flag, resolution file and host; then, per pipeline, `fit_computed` per cutoff carrying the import shas and environment lock, `predictions_computed` per eval window, and one `score_recorded` keyed by the prediction ids.
2. `lab run $D` again writes nothing: every fit and prediction is reused and every score already exists, and the report says so. Adding a pipeline to the module fits only the new one; the other pipelines cost nothing.
3. `sql "SELECT p.name, s.metric, s.value FROM latest_score l JOIN aggregate_score s ON s.score = l.score JOIN pipeline p ON p.id = l.pipeline WHERE s.window = '1'"` is the board.

Log after: one evaluation, N pipelines, one run, N scores, fits, predictions.

Done when: the second run prints zero fits and zero predictions and the query lists every pipeline.

## S3 (Later): seat the incumbent and decide

Context: one of the pipelines is the model in production.

1. `fy decide $D ridge_3m --kind promote --why "incumbent"` appends a decision on that pipeline's score with nothing to compare against. The baseline view resolves to it.
2. `fy board $D` reads each pipeline's latest score against the baseline: Pareto verdict, relative delta per metric. Reading writes nothing.
3. `fy decide $D bonsai_lw --kind promote|reject --why "..."` appends a decision carrying the baseline score, verdict and deltas it saw.

Done when: `fy history $D` shows the promotes in order, each with what it was made against, and the old baseline reads as superseded.

## S4: a scoring idea

Context: the cost assumption in the simulator changes.

1. Edit the scorer config in `experiment.py`; the evaluation id changes.
2. `lab run $D`: a new evaluation, zero fits, every pipeline rescored from its memoized predictions.
3. Scores under the old evaluation are untouched; the two evaluations never compare across.

Done when: both evaluations read from `aggregate_score` and nothing was overwritten.

## S5: code evolves

Context: a feature in `steps.py` changes.

1. `lab run $D`, committed or not. Each fit recorded the git blob sha of every module it imported; the ones whose shas no longer match refit, their predictions change, and each affected pipeline gets a new score. A dirty tree is recorded with its diff. The rest reuse.
2. `latest_score` moves to the new scores; the old ones stand in `score`, pointing at their run, commit and shas.

Done when: `sql "SELECT * FROM score WHERE pipeline = ..."` shows both scores with different runs.

## S6 (Later): iteration 100

Context: three months in.

1. The board lists every pipeline's latest score with status, verdict, deltas, aggregate metrics and the last recorded reason; history prints the promotes in order.
2. To reproduce a score: the score names its run and predictions, the run its commit and resolution file, each fit its import shas, lock and model sha. Check out, sync the lock, load the dataset blob, rerun, compare shas.
3. For analytics, attach the log in DuckDB and query the same views, or load prediction blobs by sha for per-row tests.

Done when: a newcomer can say what was tried, why each one lost, what moved the baseline, and can rebuild any number, from the log and git alone. Today the first two hold through SQL; the third needs decisions.

## S7: a pipeline fails

Context: a new pipeline raises at its third fit.

1. `lab run $D` records the two fits it finished, appends `pipeline_failed` with the traceback, runs every other pipeline, and exits 1 naming the failure. `sql "SELECT * FROM failure"` shows the error; the traceback is the blob it names.
2. Fix the code, `lab run $D`: the two recorded fits are reused, the rest are computed, the score is scored.

Done when: the second run's fit count is the fold count minus two.
