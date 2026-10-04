"""Runs: split a session, memoize fits and predictions, record a score per pipeline.

A run owns ranges, the clock and the scorer. A pipeline only sees ``fit``, ``predict``,
``save`` and ``load``. Reading is SQL over the views in ``ml_lab.ledger``.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> report.fits_computed, report.scores_recorded  # doctest: +SKIP
(9, 1)
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import dataset, formats, identity
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.session import Range, Session
from ml_lab.splits import Fold


@dataclasses.dataclass
class RunReport:
    """What a run wrote; ``run`` is empty when every fit, prediction and score already
    existed.
    """

    run: str = ""
    fits_computed: int = 0
    predictions_computed: int = 0
    scores_recorded: int = 0
    fits_reused: int = 0
    predictions_reused: int = 0
    scores_reused: int = 0
    failed: dict[str, str] = dataclasses.field(default_factory=dict)


# Public Functions =====================================================================


def run(
    ledger: Ledger,
    pipelines: list[Pipeline],
    evaluation: Evaluation,
    *,
    code_root: pathlib.Path | None = None,
    log: Callable[[str], None] = lambda line: None,
) -> RunReport:
    """Fit, predict and score each pipeline, reusing what the memo rule allows.

    ``run_started`` is appended only when something is computed, right before the first
    write, so a rerun of an unchanged tree writes nothing and adding one pipeline costs
    only its own fits, predictions and score. A pipeline that raises is recorded as
    ``pipeline_failed`` and the others continue; what it had written stays and a rerun
    resumes from there. Two names on one declaration are refused, since the ledger
    keeps one name per id and would score the second as a duplicate. ``log`` receives
    one line per fit, prediction set and score as it is written.
    """
    if not pipelines:
        raise Refused("no pipelines declared")
    names: dict[str, str] = {}
    for pipeline in pipelines:
        name = pipeline.name or pipeline.id
        if names.setdefault(pipeline.id, name) != name:
            raise Refused(
                f"pipelines {names[pipeline.id]} and {name} declare the same steps and "
                f"config, one id {pipeline.id}; rename or change one"
            )
        if pipeline.format is None:
            raise Refused(
                f"pipeline {pipeline.name or pipeline.id}: save step declares no format"
            )
    session = dataset.load(ledger, evaluation.dataset)
    schedule = evaluation.split.folds(session)
    _declare_evaluation(ledger, evaluation, schedule)
    root = code_root or identity.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
    report = RunReport()

    def start() -> str:
        if not report.run:
            report.run = identity.ulid()
            ledger.append(
                Event.RUN,
                evaluation.id,
                report.run,
                {
                    "git": _git(ledger, root),
                    "resolution": _resolution(ledger, root),
                    "host": _host(),
                    "pipelines": [{"id": p.id, "name": p.name} for p in pipelines],
                },
                id=report.run,
            )
        return report.run

    for pipeline in pipelines:
        _declare_pipeline(ledger, pipeline)
        try:
            _run_pipeline(
                ledger,
                session,
                schedule,
                pipeline,
                evaluation,
                root,
                report,
                start,
                log,
            )
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = f"{type(error).__name__}: {error}"
            ledger.append(
                Event.FAILED,
                evaluation.id,
                pipeline.id,
                {
                    "run": start(),
                    "pipeline": pipeline.id,
                    "error": report.failed[name],
                    "traceback": _text_blob(ledger, traceback.format_exc()),
                },
            )
    return report


# Running one pipeline =================================================================


def _run_pipeline(
    ledger: Ledger,
    session: Session,
    folds: list[Fold],
    pipeline: Pipeline,
    evaluation: Evaluation,
    root: pathlib.Path,
    report: RunReport,
    start: Callable[[], str],
    log: Callable[[str], None],
):
    name = pipeline.name or pipeline.id
    shas = identity.import_shas(pipeline.steps, root)
    dists = identity.imported_dists(pipeline.steps, root)
    env_lock = _text_blob(ledger, _lock_text(dists))
    loaded: set[str] | None = set(sys.modules)
    models: dict[str, Any] = {}
    predictions: dict[str, str] = {}

    def ensure_fit(fold: Fold) -> str:
        train = [list(seg) for seg in fold.train]
        fit_id = identity.content_hash(
            {
                "dataset": evaluation.dataset,
                "pipeline": pipeline.id,
                "train": train,
                "import_shas": shas,
                "env_lock": env_lock["sha"],
            }
        )
        if ledger.latest(Event.FIT, fit_id) is not None:
            report.fits_reused += 1
            return fit_id
        run_id = start()
        started = time.perf_counter()
        model = pipeline.fit(session, fold.train, pipeline.config)
        duration = time.perf_counter() - started
        models[fit_id] = model
        ledger.append(
            Event.FIT,
            evaluation.dataset,
            fit_id,
            {
                "pipeline": pipeline.id,
                "run": run_id,
                "train": train,
                "label": fold.label,
                "import_shas": shas,
                "env_lock": env_lock,
                "duration_s": duration,
                "model": {
                    "sha": ledger.put_blob(pipeline.save(model)),
                    "format": pipeline.format,
                },
            },
            id=fit_id,
        )
        report.fits_computed += 1
        log(f"fit {name} {fold.label} {duration:.1f}s")
        return fit_id

    def ensure_predictions(fit_id: str, index: int, window: str, rng: Range) -> str:
        pred_id = identity.content_hash({"fit": fit_id, "range": list(rng)})
        if ledger.latest(Event.PREDICTIONS, pred_id) is not None:
            report.predictions_reused += 1
            return pred_id
        start()
        if fit_id not in models:
            fit = ledger.latest(Event.FIT, fit_id)
            models[fit_id] = pipeline.load(
                ledger.get_blob(fit["payload"]["model"]["sha"])
            )
        pred = np.asarray(
            pipeline.predict(models[fit_id], session, rng), dtype=np.float64
        )
        ledger.append(
            Event.PREDICTIONS,
            evaluation.dataset,
            pred_id,
            {
                "fit": fit_id,
                "range": list(rng),
                "fold": index,
                "window": window,
                "blob": {
                    "sha": ledger.put_blob(formats.series_save(pred)),
                    "format": formats.Format.PARQUET,
                },
            },
            id=pred_id,
        )
        report.predictions_computed += 1
        log(f"predictions {name} {window} rows {rng[0]}:{rng[1]}")
        return pred_id

    for index, fold in enumerate(folds):
        computed_before = report.fits_computed
        fit_id = ensure_fit(fold)
        if report.fits_computed > computed_before and loaded is not None:
            _refuse_lazy_imports(pipeline, root, shas, dists, loaded)
            loaded = None
        for window, rng in fold.windows.items():
            predictions[f"{index}:{window}"] = ensure_predictions(
                fit_id, index, window, rng
            )
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
        }
    )
    if ledger.latest(Event.SCORE, score_id) is not None:
        report.scores_reused += 1
        return
    per_fold, series = [], {}
    aggregate: dict[str, dict[str, float]] = {}
    for index, fold in enumerate(folds):
        for window, rng in fold.windows.items():
            pred = _load_predictions(ledger, predictions[f"{index}:{window}"])
            rows = np.asarray(evaluation.scorer(pred, session, rng, evaluation.config))
            series.setdefault(window, []).append(rows)
            per_fold.append(
                {
                    "fold": index,
                    "label": fold.label,
                    "window": window,
                    "metrics": evaluation.metrics(rows),
                }
            )
    for window, parts in series.items():
        aggregate[window] = evaluation.metrics(np.concatenate(parts))
    ledger.append(
        Event.SCORE,
        evaluation.id,
        score_id,
        {
            "pipeline": pipeline.id,
            "run": start(),
            "predictions": predictions,
            "folds": per_fold,
            "aggregate": aggregate,
        },
        id=score_id,
    )
    report.scores_recorded += 1
    for window, metrics in aggregate.items():
        log(
            f"score {name} {window} "
            + " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        )


def _load_predictions(ledger: Ledger, pred_id: str) -> np.ndarray:
    event = ledger.latest(Event.PREDICTIONS, pred_id)
    return formats.series_load(ledger.get_blob(event["payload"]["blob"]["sha"]))


# Declarations and provenance ==========================================================


def _declare_pipeline(ledger: Ledger, pipeline: Pipeline):
    ledger.append(
        Event.PIPELINE,
        pipeline.id,
        pipeline.id,
        {
            "name": pipeline.name,
            "declaration": identity.canonical(pipeline),
            "config": identity.canonical(pipeline.config),
        },
        id=pipeline.id,
    )


def _declare_evaluation(ledger: Ledger, evaluation: Evaluation, folds: list[Fold]):
    ledger.append(
        Event.EVALUATION,
        evaluation.id,
        evaluation.id,
        {
            "dataset": evaluation.dataset,
            "declaration": identity.canonical(evaluation),
            "metrics": evaluation.directions,
            "folds": [
                {
                    "index": i,
                    "label": f.label,
                    "train": [list(seg) for seg in f.train],
                    "windows": f.windows,
                }
                for i, f in enumerate(folds)
            ],
        },
        id=evaluation.id,
    )


def _git(ledger: Ledger, root: pathlib.Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "diff": None}
    dirty = bool((git("status", "--porcelain") or "").strip())
    diff = None
    if dirty:
        sha = ledger.put_blob((git("diff", "HEAD") or "").encode())
        diff = {"sha": sha, "format": formats.Format.DIFF}
    return {"commit": commit.strip(), "dirty": dirty, "diff": diff}


def _refuse_lazy_imports(
    pipeline: Pipeline,
    root: pathlib.Path,
    shas: dict[str, str],
    dists: dict[str, str],
    loaded: set,
):
    """A fit that imports inside the function hides code from the memo; refuse and name
    it.
    """
    grown = sorted(set(identity.import_shas(pipeline.steps, root)) - set(shas))
    owners = identity.distribution_owners()
    tops = {name.partition(".")[0] for name in set(sys.modules) - loaded}
    lazy = sorted({d for top in tops for d in owners.get(top, ()) if d not in dists})
    if grown or lazy:
        raise Refused(
            f"fit imported lazily: modules {grown}, distributions {lazy}; "
            "import at module level"
        )


def _lock_text(dists: dict[str, str]) -> str:
    lines = [
        f"python=={platform.python_version()}",
        *(f"{n}=={v}" for n, v in dists.items()),
    ]
    return "\n".join(lines) + "\n"


def _resolution(ledger: Ledger, root: pathlib.Path) -> dict[str, str] | None:
    for path in [root / "uv.lock", *sorted(root.glob("requirements*.txt"))]:
        if path.exists():
            return {"path": path.name, **_text_blob(ledger, path.read_text())}
    return None


def _text_blob(ledger: Ledger, text: str) -> dict[str, str]:
    return {"sha": ledger.put_blob(text.encode()), "format": formats.Format.TEXT}


def _host() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "python": platform.python_version(),
    }
