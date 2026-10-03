"""The harness: expands a schedule, memoizes fits and predictions, scores, records decisions.

The harness owns ranges, the clock and the scorer. A pipeline only sees
``fit(session, train_range, config)`` and ``predict(model, session, range)``. Nothing here moves
a baseline on its own: the baseline is the latest promote decision, and ``decide`` records what
the decision was made against.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> decide(ledger, pipeline.id, evaluation, kind="promote", why="incumbent")  # doctest: +SKIP
'1'
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pickle
import platform
import sys
import time
from typing import Any, Final

import numpy as np

from forestry import data, hashing
from forestry.declare import Evaluation, Pipeline
from forestry.ledger import Kinds, Ledger
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
    fits_computed: int = 0
    predictions_computed: int = 0
    scored: list[str] = dataclasses.field(default_factory=list)


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


def run(ledger: Ledger, pipelines: list[Pipeline], evaluation: Evaluation) -> RunReport:
    """Fit, predict and score each pipeline under the evaluation; everything is memoized."""
    _store_evaluation(ledger, evaluation)
    session = data.session(ledger, evaluation.dataset)
    folds = expand(evaluation, session)
    report = RunReport()
    for pipeline in pipelines:
        _store_pipeline(ledger, pipeline)
        report.scored.append(pipeline.id)
        if score(ledger, pipeline.id, evaluation) is not None:
            continue
        per_fold: list[dict[str, Any]] = []
        series: dict[int, list[np.ndarray]] = {}
        for fold in folds:
            fit_id = _ensure_fit(ledger, session, evaluation, pipeline, fold, report)
            for age, rng in fold.evals.items():
                pred = _ensure_predictions(
                    ledger, session, pipeline, fit_id, rng, fold, age, report
                )
                rows = np.asarray(evaluation.scorer(pred, session, rng, evaluation.config))
                series.setdefault(age, []).append(rows)
                metrics = evaluation.metrics(rows)
                per_fold.append(
                    {"fold": fold.index, "cutoff": str(fold.cutoff), "age": age, **metrics}
                )
        ledger.put(
            Kinds.SCORE,
            _score_id(pipeline.id, evaluation),
            {
                "pipeline": pipeline.id,
                "evaluation": evaluation.id,
                "folds": per_fold,
                "aggregate": {
                    str(age): evaluation.metrics(np.concatenate(parts))
                    for age, parts in series.items()
                },
            },
        )
    return report


def decide(ledger: Ledger, pipeline_id: str, evaluation: Evaluation, *, kind: str, why: str) -> str:
    """Record promote or reject on a scored pipeline, with the comparison it was made against."""
    if kind not in (Decision.PROMOTE, Decision.REJECT):
        raise Refused(f"decision kind {kind!r}; promote or reject")
    if score(ledger, pipeline_id, evaluation) is None:
        raise Refused("pipeline has no score under this evaluation; run it first")
    decision_id = ledger.next_decision_id()
    ledger.put(
        Kinds.DECISION,
        decision_id,
        {
            "kind": kind,
            "why": why,
            "pipeline": pipeline_id,
            "evaluation": evaluation.id,
            "against": compare(ledger, pipeline_id, evaluation),
            "at": time.time(),
        },
    )
    return decision_id


def baseline(ledger: Ledger, evaluation: Evaluation) -> str | None:
    """The pipeline promoted last under this evaluation, or None."""
    promotions = [
        d
        for d in ledger.where(Kinds.DECISION, evaluation=evaluation.id)
        if d["kind"] == Decision.PROMOTE
    ]
    return promotions[-1]["pipeline"] if promotions else None


def compare(ledger: Ledger, pipeline_id: str, evaluation: Evaluation) -> dict[str, Any] | None:
    """Verdict and relative deltas of a pipeline against the baseline; None without a baseline."""
    current = baseline(ledger, evaluation)
    if current is None or current == pipeline_id:
        return None
    challenger = aggregate(ledger, pipeline_id, evaluation)
    incumbent = aggregate(ledger, current, evaluation)
    return {
        "baseline": current,
        "verdict": dominance(challenger, incumbent, evaluation.directions),
        "deltas": relative_deltas(challenger, incumbent, evaluation.directions),
    }


def score(ledger: Ledger, pipeline_id: str, evaluation: Evaluation) -> dict[str, Any] | None:
    return ledger.get(Kinds.SCORE, _score_id(pipeline_id, evaluation))


def aggregate(ledger: Ledger, pipeline_id: str, evaluation: Evaluation) -> dict[str, float]:
    """The aggregate metric vector at the first age."""
    row = score(ledger, pipeline_id, evaluation)
    if row is None:
        raise Refused(f"pipeline {pipeline_id} has no score under this evaluation")
    return row["aggregate"][str(evaluation.ages[0])]


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


# Private Functions ================================================================================


def _ensure_fit(
    ledger: Ledger,
    session: Session,
    evaluation: Evaluation,
    pipeline: Pipeline,
    fold: Fold,
    report: RunReport,
) -> str:
    fit_id = hashing.content_hash(
        {"dataset": evaluation.dataset, "pipeline": pipeline.id, "train": list(fold.train)}
    )
    if ledger.get(Kinds.FIT, fit_id) is not None:
        return fit_id
    started = time.perf_counter()
    model = pipeline.fit(session, fold.train, pipeline.config)
    ledger.put(
        Kinds.FIT,
        fit_id,
        {
            "dataset": evaluation.dataset,
            "pipeline": pipeline.id,
            "train": list(fold.train),
            "cutoff": str(fold.cutoff),
            "code_sha": hashing.step_ref(pipeline.fit).file_sha,
            "env_lock": _env_lock(),
            "host": _host(),
            "duration_s": time.perf_counter() - started,
            "blob": ledger.put_blob(pickle.dumps(model)),
        },
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
    report: RunReport,
) -> np.ndarray:
    pred_id = hashing.content_hash({"fit": fit_id, "range": list(rng)})
    existing = ledger.get(Kinds.PREDICTIONS, pred_id)
    if existing is not None:
        return np.frombuffer(ledger.get_blob(existing["blob"]), dtype=np.float64)
    model = pickle.loads(ledger.get_blob(ledger.get(Kinds.FIT, fit_id)["blob"]))
    pred = np.asarray(pipeline.predict(model, session, rng), dtype=np.float64)
    ledger.put(
        Kinds.PREDICTIONS,
        pred_id,
        {
            "fit": fit_id,
            "range": list(rng),
            "fold": fold.index,
            "age": age,
            "blob": ledger.put_blob(pred.tobytes()),
        },
    )
    report.predictions_computed += 1
    return pred


def _store_pipeline(ledger: Ledger, pipeline: Pipeline):
    ledger.put(
        Kinds.PIPELINE,
        pipeline.id,
        {
            "name": pipeline.name,
            "declaration": hashing.canonical(pipeline),
            "config": hashing.canonical(pipeline.config),
            "pickle": ledger.put_blob(pickle.dumps(pipeline)),
        },
    )


def _store_evaluation(ledger: Ledger, evaluation: Evaluation):
    ledger.put(
        Kinds.EVALUATION,
        evaluation.id,
        {
            "dataset": evaluation.dataset,
            "declaration": hashing.canonical(evaluation),
            "pickle": ledger.put_blob(pickle.dumps(evaluation)),
        },
    )


def _score_id(pipeline_id: str, evaluation: Evaluation) -> str:
    return hashing.content_hash({"pipeline": pipeline_id, "evaluation": evaluation.id})


def _signed(a: dict[str, float], b: dict[str, float], m: str, directions: dict[str, str]) -> float:
    sign = 1.0 if directions[m] == "max" else -1.0
    return sign * (a[m] - b[m])


def _env_lock() -> str:
    return hashing.content_hash({"python": sys.version, "numpy": np.__version__})


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
