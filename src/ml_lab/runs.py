"""Runs: split a dataset, memoize fits and predictions, record a score per pipeline.

A run owns the folds, the clock, the guards and the scorer. A pipeline's functions see
the rows their slot may read, and the scorer's ``series`` sees the fold's test range
with its predictions. Reading is SQL over the views in ``ml_lab.ledger``. The engine
for one pipeline is ``ml_lab.stages``, the refusals ``ml_lab.guards``, the declaration
events ``ml_lab.declarations`` and the run's facts ``ml_lab.provenance``.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> report.fits_computed, report.scores_recorded  # doctest: +SKIP
(9, 1)
"""

from __future__ import annotations

import dataclasses
import pathlib
import sqlite3
import sys
import traceback
from collections.abc import Callable

import numpy as np

from ml_lab import formats, identity, provenance
from ml_lab.dataset import (
    load,
    resolve,
)
from ml_lab.declarations import (
    SLOTS,
    declare_evaluation,
    declare_pipeline,
    nested,
    project_root,
    refuse_half_declared,
    refuse_unseen_class,
    scorer_functions,
    stage_functions,
)
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.guards import (
    finite_metrics,
    refuse_lazy_imports,
    refuse_overlap,
    refuse_sealed,
    refuse_unrevealed,
    series_array,
)
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.splits import Fold
from ml_lab.stages import RunContext, RunReport, load_predictions

# Public Functions ===================================================================


def run(
    ledger: Ledger,
    pipelines: list[Pipeline],
    evaluation: Evaluation,
    *,
    code_root: pathlib.Path | None = None,
    log: Callable[[str], None] = lambda line: None,
    dry: bool = False,
    planned: set[str] | None = None,
) -> RunReport:
    """Fit, predict and score each pipeline, reusing what the memo rule allows.

    ``run_started`` is appended only when something is computed, right before the first
    write, so a rerun of an unchanged tree writes nothing and adding one pipeline costs
    only its own fits, predictions and score. Every function sees a prefix of the
    dataset: ``fit`` up to the end of its last train segment with every column NaN
    outside the train segments, ``predict``, ``postprocess`` and the scorer's
    ``series`` up to the end of the fold's test range with unknown labels NaN. ``log``
    receives one line when a feature set or a fit starts and one per feature set, fit,
    prediction set and score as it is written, so a log shows what is running and since
    when. A ``dry`` run computes every id, looks each one up and writes nothing; it
    logs what it would compute and, for a feature set or a fit, which inputs moved
    against the nearest earlier one of the same function or pipeline. A feature set no
    run has computed is computed and probed in a dry run too, since its ids depend on
    its columns and its probe needs no fit, so a leak it reads is refused before any
    fit is paid for, on the first probe span only to keep a dry run cheap; the full
    probes and those of ``predict`` and ``postprocess`` run only in the real run.
    ``planned`` carries the ids a dry run would write across several calls, so shared
    work counts once. ``code_root`` is where
    repo modules are hashed from, the git root of the first pipeline's fit when not
    given.

    Refused before anything is read or written, at any depth of blends inside blends:
    no pipelines; two names on one declaration; a slot holding a lambda, closure,
    method or anything that is not a module-level function; a ``format`` outside
    ``formats.KNOWN``; a params class at any depth defined in a repo module its
    function does not import; a reached module the memo cannot see
    (``identity.refuse_unseen_code``).

    Refused before anything is written: a ``dataset_id`` prefix that matches no
    recorded dataset or several; a split's own refusals (too few folds, an empty test
    range, a fold label twice, a group that recurs); a fold whose test range overlaps
    its train segments,
    whose folds train on labels not revealed when their test range starts (the message
    names the purge that clears them); a split whose folds moved under an unchanged
    declaration; a sealed evaluation on a dataset without a sealed tail, an
    evaluation that is not sealed validating on one, a sealed evaluation validating
    before it, and a fold training on a tail row no earlier fold of a sealed
    evaluation tested.

    Recorded as ``pipeline_failed``, after which the others continue and a rerun resumes
    from what was written: an error a function raises, noted with the fold's label; an
    output that is not one row per row of the range, or whose width changes between
    folds; a prediction or features column that changes when recomputed on a shorter
    prefix; a model reloaded from its saved bytes that predicts differently; a scored
    prediction or a metric that is not finite; a series that is a scalar or empty, or
    whose column count ``Scorer.columns`` disagrees with; an import inside a function
    body; a module edited during the run; a second score of one pipeline id under a
    sealed evaluation, found by computing its ids before any fit or prediction, and
    held at the write against a concurrent run.
    """
    if not pipelines:
        raise Refused("no pipelines declared")
    names: dict[str, str] = {}
    for pipeline in pipelines:
        name = pipeline.name or pipeline.id
        if names.setdefault(pipeline.id, name) != name:
            raise Refused(
                f"pipelines {names[pipeline.id]} and {name} declare the same functions "
                f"and params, one id {pipeline.id}; rename or change one"
            )
        for p in nested(pipeline):
            for slot, held in [(s, getattr(p, s)) for s in SLOTS] + [
                ("features", f) for f in p.feature_functions
            ]:
                if held is not None and not callable(held):
                    raise Refused(
                        f"pipeline {p.name or name}: {slot} holds a "
                        f"{type(held).__name__}, not a function"
                    )
        for p in nested(pipeline):
            refuse_half_declared(p, name)
    root = code_root or project_root(pipelines)
    scoring = scorer_functions(evaluation)
    identity.refuse_unseen_code(
        [
            *scoring,
            *(
                f
                for pipeline in pipelines
                for p in nested(pipeline)
                for stage in stage_functions(p).values()
                for f in stage
            ),
        ],
        root,
    )
    for pipeline in pipelines:
        for p in nested(pipeline):
            if p.fit is not None:
                model = (p.fit, p.save, p.load)
                shas = identity.import_shas([f for f in model if f is not None], root)
                refuse_unseen_class(p.params, shas, root, "fit")
            shas = identity.import_shas((p.predict,), root)
            refuse_unseen_class(p.predict_config, shas, root, "predict")
            if p.postprocess is not None:
                shas = identity.import_shas((p.postprocess,), root)
                refuse_unseen_class(p.postprocess_config, shas, root, "postprocess")
    scorer_shas = identity.import_shas(scoring[:2], root)
    refuse_unseen_class(evaluation.scorer_params, scorer_shas, root, "scorer")
    evaluation = dataclasses.replace(
        evaluation, dataset_id=resolve(ledger, evaluation.dataset_id)
    )
    dataset = load(ledger, evaluation.dataset_id)
    recorded = ledger.event(Event.DATASET, evaluation.dataset_id)["payload"]
    targets = dataset.reveal
    schedule = evaluation.split.folds(dataset)
    refuse_overlap(schedule)
    refuse_unrevealed(dataset, targets, schedule)
    refuse_sealed(evaluation, schedule, recorded["sealed_from"])
    declare_evaluation(ledger, evaluation, schedule, dry)
    report = RunReport()

    def start() -> str:
        if not report.run:
            run_id = identity.ulid()
            ledger.append(
                Event.RUN,
                evaluation.id,
                run_id,
                {
                    "git": provenance.git_state(ledger, root),
                    "resolution": provenance.resolution(ledger, root),
                    "editable": {
                        name: provenance.git_state(ledger, at)
                        for name, at in identity.editable_roots(
                            [*scoring, *(f for p in pipelines for f in p.functions)],
                            root,
                        ).items()
                    },
                    "host": provenance.host(),
                    "pipelines": [{"id": p.id, "name": p.name} for p in pipelines],
                },
                id=run_id,
            )
            report.run = run_id
        return report.run

    context = RunContext(
        ledger,
        dataset,
        targets,
        evaluation,
        root,
        report,
        start,
        log,
        dry,
        set() if planned is None else planned,
    )
    for pipeline in pipelines:
        if not dry:
            for declared in nested(pipeline):
                declare_pipeline(ledger, declared)
        try:
            if (
                evaluation.sealed
                and not dry
                and _held_score(ledger, evaluation, pipeline)
            ):
                quiet = dataclasses.replace(
                    context,
                    report=RunReport(),
                    log=lambda line: None,
                    dry=True,
                    planned=set(),
                    features={},
                    stages={},
                )
                _run_pipeline(quiet, schedule, pipeline)
            _run_pipeline(context, schedule, pipeline)
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = "; ".join(
                [f"{type(error).__name__}: {error}", *getattr(error, "__notes__", ())]
            )
            report.failed_ids.add(pipeline.id)
            context.stages.clear()
            if dry:
                continue
            ledger.append(
                Event.FAILED,
                evaluation.id,
                pipeline.id,
                {
                    "run": start(),
                    "pipeline": pipeline.id,
                    "error": report.failed[name],
                    "traceback": provenance.text_blob(ledger, traceback.format_exc()),
                },
            )
    return report


# Private Functions ==================================================================


def _run_pipeline(context: RunContext, folds: list[Fold], pipeline: Pipeline):
    stage = context.stage(pipeline)
    predictions: dict[str, str] = {}
    for index, fold in enumerate(folds):
        fit_id = stage.fit(fold, index)
        predictions[str(index)] = stage.predictions(fit_id, fold, index, fold.test)
    _score(context, folds, pipeline, predictions)


def _score(
    context: RunContext,
    folds: list[Fold],
    pipeline: Pipeline,
    predictions: dict[str, str],
):
    ledger, evaluation, report = context.ledger, context.evaluation, context.report
    name = pipeline.name or pipeline.id
    scorer = evaluation.scorer
    scoring = scorer_functions(evaluation)
    scorer_shas = identity.import_shas(scoring, context.root)
    dists = identity.imported_dists(scoring, context.root)
    lock = provenance.lock_text(dists)
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
            "scorer_keys": identity.code_keys(scoring, context.root),
            "env_lock": identity.bytes_hash(lock.encode()),
            "directions": evaluation.scorer.directions,
        }
    )
    if ledger.latest(Event.SCORE, score_id) is not None:
        report.scores_reused += 1
        return
    if evaluation.sealed:
        _refuse_rescoring(ledger, evaluation, pipeline)
    if context.dry:
        report.scores_recorded += 1
        context.log(f"would score {name}")
        return
    per_fold, parts = [], []
    loaded = set(sys.modules)
    for index, fold in enumerate(folds):
        pred = load_predictions(ledger, predictions[str(index)])
        visible = context.dataset.upto(fold.test[1])
        series = series_array(
            f"{name}: the scorer's series on fold {fold.label} has",
            scorer.series(pred, visible, fold.test, evaluation.scorer_params),
        )
        parts.append(series)
        metrics = finite_metrics(
            name, evaluation.scorer.metrics(series.copy()), f"fold {fold.label}"
        )
        per_fold.append({"fold": index, "label": fold.label, "metrics": metrics})
    whole = np.concatenate(parts)
    aggregate = finite_metrics(
        name, evaluation.scorer.metrics(whole.copy()), "the pooled folds"
    )
    refuse_lazy_imports(pipeline, context.root, dists, loaded, "scorer")
    columns = np.asarray(whole, np.float64).reshape(len(whole), -1)
    names = scorer.columns or [str(i) for i in range(columns.shape[1])]
    if len(names) != columns.shape[1]:
        raise Refused(
            f"scorer declares columns {list(names)}; its series has {columns.shape[1]}"
        )
    stored = {
        "sha": ledger.put_blob(
            formats.arrays_save(dict(zip(names, columns.T, strict=True)))
        ),
        "format": formats.Format.ARROW_ARRAYS,
        "fold_rows": [len(part) for part in parts],
    }
    try:
        ledger.append(
            Event.SCORE,
            evaluation.id,
            score_id,
            {
                "pipeline": pipeline.id,
                "run": context.start(),
                "predictions": predictions,
                "folds": per_fold,
                "aggregate": aggregate,
                "directions": evaluation.scorer.directions,
                "series": stored,
                "scorer_shas": scorer_shas,
                "sealed": evaluation.sealed,
            },
            id=score_id,
        )
    except sqlite3.IntegrityError:
        _refuse_rescoring(ledger, evaluation, pipeline)
        raise
    report.scores_recorded += 1
    context.log(
        f"score {name} " + " ".join(f"{k}={v:.4g}" for k, v in aggregate.items())
    )


def _held_score(ledger: Ledger, evaluation: Evaluation, pipeline: Pipeline) -> str:
    held = ledger.sql(
        "SELECT id FROM event_score WHERE evaluation = ? AND pipeline = ?",
        (evaluation.id, pipeline.id),
    )
    return held[0]["id"] if held else ""


def _refuse_rescoring(ledger: Ledger, evaluation: Evaluation, pipeline: Pipeline):
    held = _held_score(ledger, evaluation, pipeline)
    if held:
        raise Refused(
            f"sealed evaluation {evaluation.name or evaluation.id} already scored "
            f"pipeline {pipeline.name or pipeline.id} as score {held}; a "
            "sealed evaluation scores each pipeline once"
        )
