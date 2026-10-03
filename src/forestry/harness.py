"""The harness: expands a schedule, memoizes fits and predictions, scores, records decisions.

The harness owns ranges, the clock and the scorer. A pipeline only sees ``fit``, ``predict``,
``save`` and ``load``. Nothing here moves a baseline on its own: the baseline is the latest
promote decision, and ``decide`` records what the decision was made against.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> decide(ledger, pipeline.id, evaluation, kind="promote", why="incumbent")  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Final

import numpy as np

from forestry import data, formats, hashing
from forestry.declare import Evaluation, Pipeline
from forestry.ledger import Event, Ledger
from forestry.session import Range, Session, add_months, as_date


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


class Verdict:
    DOMINATES: Final = "dominates"
    DOMINATED: Final = "dominated"
    INCOMPARABLE: Final = "incomparable"


class Decision:
    PROMOTE: Final = "promote"
    REJECT: Final = "reject"


@dataclasses.dataclass
class Fold:
    index: int
    cutoff: datetime.date
    train: Range
    evals: dict[int, Range]


@dataclasses.dataclass
class RunReport:
    run: str = ""
    fits_computed: int = 0
    predictions_computed: int = 0
    entries_scored: int = 0
    pipelines: list[str] = dataclasses.field(default_factory=list)


# Public Functions =================================================================================


def expand(evaluation: Evaluation, session: Session) -> list[Fold]:
    """Folds from the schedule: every cutoff whose oldest eval window still fits the data."""
    span = max(evaluation.ages) * evaluation.eval_months
    folds: list[Fold] = []
    cutoff = as_date(evaluation.first_cutoff)
    while add_months(cutoff, span) <= session.end_exclusive:
        train = (0, session.index_of(cutoff, -evaluation.embargo_seconds))
        evals = {
            age: (
                session.index_of(add_months(cutoff, (age - 1) * evaluation.eval_months)),
                session.index_of(add_months(cutoff, age * evaluation.eval_months)),
            )
            for age in evaluation.ages
        }
        folds.append(Fold(len(folds), cutoff, train, evals))
        cutoff = add_months(cutoff, evaluation.every_months)
    if len(folds) < evaluation.min_folds:
        raise Refused(f"schedule yields {len(folds)} folds, min_folds is {evaluation.min_folds}")
    return folds


def run(
    ledger: Ledger,
    pipelines: list[Pipeline],
    evaluation: Evaluation,
    *,
    code_root: pathlib.Path | None = None,
) -> RunReport:
    """Post a run: fit, predict and score each pipeline, reusing what the memo rule allows."""
    for pipeline in pipelines:
        if pipeline.format is None:
            raise Refused(f"pipeline {pipeline.name or pipeline.id}: save step declares no format")
    session = data.session(ledger, evaluation.dataset)
    folds = expand(evaluation, session)
    _declare_evaluation(ledger, evaluation, folds)
    root = code_root or hashing.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
    env_lock = _env_lock()
    report = RunReport(run=hashing.ulid())
    ledger.append(
        Event.RUN,
        evaluation.id,
        report.run,
        {
            "git": _git(ledger, root),
            "env_lock": env_lock,
            "host": _host(),
            "pipelines": [{"id": p.id, "name": p.name} for p in pipelines],
        },
        id=report.run,
    )
    for pipeline in pipelines:
        _declare_pipeline(ledger, pipeline)
        report.pipelines.append(pipeline.id)
        _run_pipeline(ledger, session, folds, pipeline, evaluation, root, env_lock, report)
    return report


def decide(ledger: Ledger, pipeline_id: str, evaluation: Evaluation, *, kind: str, why: str) -> str:
    """Record promote or reject on a pipeline's latest entry, with what it was made against."""
    if kind not in (Decision.PROMOTE, Decision.REJECT):
        raise Refused(f"decision kind {kind!r}; promote or reject")
    current = latest_entry(ledger, pipeline_id, evaluation)
    if current is None:
        raise Refused("pipeline has no entry under this evaluation; run it first")
    decision_id = hashing.ulid()
    ledger.append(
        Event.DECISION,
        evaluation.id,
        current["id"],
        {
            "kind": kind,
            "why": why,
            "pipeline": pipeline_id,
            "against": compare(ledger, pipeline_id, evaluation),
        },
        id=decision_id,
    )
    return decision_id


def baseline(ledger: Ledger, evaluation: Evaluation, upto: int | None = None) -> str | None:
    """The entry promoted last under this evaluation, or None."""
    promotions = [
        d
        for d in ledger.events(Event.DECISION, stream=evaluation.id, upto=upto)
        if d["payload"]["kind"] == Decision.PROMOTE
    ]
    return promotions[-1]["key"] if promotions else None


def latest_entry(
    ledger: Ledger, pipeline_id: str, evaluation: Evaluation, upto: int | None = None
) -> dict[str, Any] | None:
    """The newest entry event for a pipeline under an evaluation, or None."""
    entries = [
        e
        for e in ledger.events(Event.ENTRY, stream=evaluation.id, upto=upto)
        if e["payload"]["pipeline"] == pipeline_id
    ]
    return entries[-1] if entries else None


def aggregate(ledger: Ledger, entry_id: str, evaluation: Evaluation) -> dict[str, float]:
    """An entry's aggregate metric vector at the first age."""
    event = ledger.get(entry_id)
    if event is None or event["type"] != Event.ENTRY:
        raise Refused(f"no entry {entry_id}")
    return event["payload"]["aggregate"][str(evaluation.ages[0])]


def compare(
    ledger: Ledger, pipeline_id: str, evaluation: Evaluation, upto: int | None = None
) -> dict[str, Any] | None:
    """Verdict and relative deltas of a pipeline's latest entry against the baseline entry."""
    head = baseline(ledger, evaluation, upto)
    current = latest_entry(ledger, pipeline_id, evaluation, upto)
    if head is None or current is None or head == current["id"]:
        return None
    challenger = aggregate(ledger, current["id"], evaluation)
    incumbent = aggregate(ledger, head, evaluation)
    return {
        "entry": head,
        "pipeline": ledger.get(head)["payload"]["pipeline"],
        "verdict": dominance(challenger, incumbent, evaluation.directions),
        "deltas": relative_deltas(challenger, incumbent, evaluation.directions),
    }


def dominance(a: dict[str, float], b: dict[str, float], directions: dict[str, str]) -> str:
    """Pareto verdict of ``a`` against ``b``: dominates, dominated or incomparable."""
    deltas = [_signed(a, b, m, directions) for m in directions]
    if all(d >= 0 for d in deltas) and any(d > 0 for d in deltas):
        return Verdict.DOMINATES
    if all(d <= 0 for d in deltas) and any(d < 0 for d in deltas):
        return Verdict.DOMINATED
    return Verdict.INCOMPARABLE


def relative_deltas(
    a: dict[str, float], b: dict[str, float], directions: dict[str, str]
) -> dict[str, float]:
    """Per metric, how much better ``a`` is than ``b``, relative to ``b``; positive is better."""
    return {m: _signed(a, b, m, directions) / max(abs(b[m]), 1e-12) for m in directions}


# Running one pipeline =============================================================================


def _run_pipeline(
    ledger: Ledger,
    session: Session,
    folds: list[Fold],
    pipeline: Pipeline,
    evaluation: Evaluation,
    root: pathlib.Path,
    env_lock: str,
    report: RunReport,
):
    shas = hashing.import_shas(pipeline.steps, root)
    models: dict[str, Any] = {}
    predictions: dict[str, str] = {}
    for fold in folds:
        fit_id = _ensure_fit(
            ledger, session, evaluation, pipeline, fold, shas, env_lock, models, report
        )
        if report.fits_computed and shas != hashing.import_shas(pipeline.steps, root):
            grown = sorted(set(hashing.import_shas(pipeline.steps, root)) - set(shas))
            raise Refused(f"fit imported repo modules lazily: {grown}; import them at module level")
        for age, rng in fold.evals.items():
            predictions[f"{fold.index}:{age}"] = _ensure_predictions(
                ledger, session, pipeline, fit_id, rng, fold, age, models, report
            )
    entry_id = hashing.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
        }
    )
    if ledger.latest(Event.ENTRY, entry_id) is not None:
        return
    per_fold, series = [], {}
    for fold in folds:
        for age, rng in fold.evals.items():
            pred = _load_predictions(ledger, predictions[f"{fold.index}:{age}"])
            rows = np.asarray(evaluation.scorer(pred, session, rng, evaluation.config))
            series.setdefault(age, []).append(rows)
            per_fold.append(
                {
                    "fold": fold.index,
                    "cutoff": str(fold.cutoff),
                    "age": age,
                    "metrics": evaluation.metrics(rows),
                }
            )
    ledger.append(
        Event.ENTRY,
        evaluation.id,
        entry_id,
        {
            "pipeline": pipeline.id,
            "run": report.run,
            "predictions": predictions,
            "folds": per_fold,
            "aggregate": {
                str(age): evaluation.metrics(np.concatenate(parts)) for age, parts in series.items()
            },
        },
        id=entry_id,
    )
    report.entries_scored += 1


def _ensure_fit(
    ledger: Ledger,
    session: Session,
    evaluation: Evaluation,
    pipeline: Pipeline,
    fold: Fold,
    shas: dict[str, str],
    env_lock: str,
    models: dict[str, Any],
    report: RunReport,
) -> str:
    fit_id = hashing.content_hash(
        {
            "dataset": evaluation.dataset,
            "pipeline": pipeline.id,
            "train": list(fold.train),
            "import_shas": shas,
            "env_lock": env_lock,
        }
    )
    if ledger.latest(Event.FIT, fit_id) is not None:
        return fit_id
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
            "run": report.run,
            "train": list(fold.train),
            "cutoff": str(fold.cutoff),
            "import_shas": shas,
            "env_lock": env_lock,
            "duration_s": duration,
            "model": {"sha": ledger.put_blob(pipeline.save(model)), "format": pipeline.format},
        },
        id=fit_id,
    )
    report.fits_computed += 1
    return fit_id


def _ensure_predictions(
    ledger: Ledger,
    session: Session,
    pipeline: Pipeline,
    fit_id: str,
    rng: Range,
    fold: Fold,
    age: int,
    models: dict[str, Any],
    report: RunReport,
) -> str:
    pred_id = hashing.content_hash({"fit": fit_id, "range": list(rng)})
    if ledger.latest(Event.PREDICTIONS, pred_id) is not None:
        return pred_id
    if fit_id not in models:
        fit = ledger.latest(Event.FIT, fit_id)
        models[fit_id] = pipeline.load(ledger.get_blob(fit["payload"]["model"]["sha"]))
    pred = np.asarray(pipeline.predict(models[fit_id], session, rng), dtype=np.float64)
    ledger.append(
        Event.PREDICTIONS,
        ledger.latest(Event.FIT, fit_id)["stream"],
        pred_id,
        {
            "fit": fit_id,
            "range": list(rng),
            "fold": fold.index,
            "age": age,
            "blob": {
                "sha": ledger.put_blob(formats.series_save(pred)),
                "format": formats.Format.PARQUET,
            },
        },
        id=pred_id,
    )
    report.predictions_computed += 1
    return pred_id


def _load_predictions(ledger: Ledger, pred_id: str) -> np.ndarray:
    event = ledger.latest(Event.PREDICTIONS, pred_id)
    return formats.series_load(ledger.get_blob(event["payload"]["blob"]["sha"]))


# Declarations and provenance ======================================================================


def _declare_pipeline(ledger: Ledger, pipeline: Pipeline):
    ledger.append(
        Event.PIPELINE,
        pipeline.id,
        pipeline.id,
        {
            "name": pipeline.name,
            "declaration": hashing.canonical(pipeline),
            "config": hashing.canonical(pipeline.config),
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
            "declaration": hashing.canonical(evaluation),
            "metrics": evaluation.directions,
            "folds": [
                {
                    "index": f.index,
                    "cutoff": str(f.cutoff),
                    "train": list(f.train),
                    "windows": f.evals,
                }
                for f in folds
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


def _env_lock() -> str:
    dists = sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()})
    return hashing.content_hash({"python": sys.version, "dists": [list(d) for d in dists]})


def _host() -> dict[str, Any]:
    cpu_max = None
    try:
        cpu_max = open("/sys/fs/cgroup/cpu.max").read().strip()
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "cgroup_cpu_max": cpu_max,
        "python": platform.python_version(),
    }


def _signed(a: dict[str, float], b: dict[str, float], m: str, directions: dict[str, str]) -> float:
    sign = 1.0 if directions[m] == "max" else -1.0
    return sign * (a[m] - b[m])
