"""The harness: expands a schedule, memoizes fits and predictions, scores, compares, records.

The harness owns ranges, the clock and the scorer. A pipeline only sees
``fit(session, train_range, config)`` and ``predict(model, session, range)``. Nothing here
moves a baseline on its own: ``compare`` writes a verdict, ``decide`` writes a person's or an
agent's choice.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> compare(ledger, report.candidates[0], evaluation)["verdict"]  # doctest: +SKIP
'dominates'
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pickle
import platform
import sys
import time
import uuid
from typing import Any, Final

import numpy as np

from forestry import data, hashing
from forestry.declare import Evaluation, Pipeline
from forestry.ledger import Kinds, Ledger
from forestry.session import Range, Session, add_months, as_date


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


class Status:
    SCORED: Final = "scored"
    BASELINE: Final = "baseline"
    SUPERSEDED: Final = "superseded"
    REJECTED: Final = "rejected"


class Verdict:
    DOMINATES: Final = "dominates"
    DOMINATED: Final = "dominated"
    INCOMPARABLE: Final = "incomparable"


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
    candidates: list[str] = dataclasses.field(default_factory=list)


# Public Functions =================================================================================


def expand(evaluation: Evaluation, session: Session) -> list[Fold]:
    """Folds from the schedule: every cutoff whose oldest eval window still fits the data."""
    schedule = evaluation.schedule
    span = max(schedule.ages) * schedule.eval_months
    folds: list[Fold] = []
    cutoff = as_date(schedule.first_cutoff)
    while add_months(cutoff, span) <= session.frame.end_exclusive:
        train = (0, session.index_of(cutoff, -schedule.embargo_seconds))
        evals = {
            age: (
                session.index_of(add_months(cutoff, (age - 1) * schedule.eval_months)),
                session.index_of(add_months(cutoff, age * schedule.eval_months)),
            )
            for age in schedule.ages
        }
        folds.append(Fold(len(folds), cutoff, train, evals))
        cutoff = add_months(cutoff, schedule.every_months)
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
        candidate = _ensure_candidate(ledger, evaluation, pipeline.id)
        report.candidates.append(candidate)
        if ledger.where(Kinds.SCORE, candidate=candidate, fold=None):
            continue
        by_age: dict[int, list[Any]] = {}
        for fold in folds:
            fit_id = _ensure_fit(ledger, session, evaluation, pipeline, fold, report)
            for age, rng in fold.evals.items():
                pred = _ensure_predictions(
                    ledger, session, pipeline, fit_id, rng, fold, age, report
                )
                result = evaluation.scorer(pred, session, rng, evaluation.config)
                by_age.setdefault(age, []).append(result)
                _put_score(ledger, evaluation, candidate, fold.index, age, result.metrics)
        for age, results in by_age.items():
            metrics = _aggregate_results(evaluation, results)
            _put_score(ledger, evaluation, candidate, None, age, metrics)
    return report


def compare(ledger: Ledger, candidate_id: str, evaluation: Evaluation) -> dict[str, Any]:
    """Compare a candidate's aggregate against the baseline's; writes and returns the row."""
    cand = _candidate(ledger, candidate_id)
    if cand["evaluation"] != evaluation.id:
        raise Refused("candidate was scored under a different evaluation")
    current = ledger.get(Kinds.BASELINE, evaluation.id)
    if current is None:
        raise Refused("no baseline under this evaluation; promote one first")
    if _fold_count(ledger, candidate_id) < evaluation.min_folds:
        raise Refused("fewer folds than min_folds")
    challenger = aggregate(ledger, candidate_id, evaluation.compare_age)
    incumbent = aggregate(ledger, current["candidate"], evaluation.compare_age)
    row = {
        "challenger": candidate_id,
        "baseline": current["candidate"],
        "evaluation": evaluation.id,
        "verdict": dominance(challenger, incumbent, evaluation.directions),
        "deltas": relative_deltas(challenger, incumbent, evaluation.directions),
        "challenger_metrics": challenger,
        "baseline_metrics": incumbent,
        "decision": None,
    }
    comparison_id = uuid.uuid4().hex
    ledger.put(Kinds.COMPARISON, comparison_id, row)
    return {"id": comparison_id, **row}


def decide(ledger: Ledger, candidate_id: str, *, kind: str, why: str) -> str:
    """Record a decision on a candidate: promote moves the baseline, reject closes it."""
    cand = _candidate(ledger, candidate_id)
    if kind not in ("promote", "reject"):
        raise Refused(f"decision kind {kind!r}; promote or reject")
    comparison = _latest_comparison(ledger, candidate_id)
    decision = _decision(ledger, kind=kind, why=why, candidate=candidate_id, comparison=comparison)
    if comparison is not None:
        ledger.update(Kinds.COMPARISON, comparison, decision=decision)
    if kind == "promote":
        _set_baseline(ledger, cand["evaluation"], candidate_id, decision)
        ledger.update(Kinds.CANDIDATE, candidate_id, status=Status.BASELINE, reason=why)
    else:
        ledger.update(Kinds.CANDIDATE, candidate_id, status=Status.REJECTED, reason=why)
    return decision


def aggregate(ledger: Ledger, candidate_id: str, age: int) -> dict[str, float]:
    """The aggregate metric vector of a candidate at one age."""
    rows = [
        s for s in ledger.where(Kinds.SCORE, candidate=candidate_id, fold=None) if s["age"] == age
    ]
    if not rows:
        raise Refused(f"candidate {candidate_id} has no aggregate score at age {age}")
    return rows[0]["metrics"]


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


# Fits and predictions =============================================================================


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
    duration = time.perf_counter() - started
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
            "duration_s": duration,
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
            "pipeline": pipeline.id,
            "blob": ledger.put_blob(pred.tobytes()),
        },
    )
    report.predictions_computed += 1
    return pred


# Rows =============================================================================================


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


def _ensure_candidate(ledger: Ledger, evaluation: Evaluation, pipeline_id: str) -> str:
    key = {"dataset": evaluation.dataset, "pipeline": pipeline_id, "evaluation": evaluation.id}
    cand_id = hashing.content_hash(key)
    ledger.put(Kinds.CANDIDATE, cand_id, {**key, "status": Status.SCORED, "reason": None})
    return cand_id


def _put_score(
    ledger: Ledger,
    evaluation: Evaluation,
    candidate: str,
    fold: int | None,
    age: int,
    metrics: dict[str, float],
):
    ledger.put(
        Kinds.SCORE,
        hashing.content_hash({"candidate": candidate, "fold": fold, "age": age}),
        {
            "candidate": candidate,
            "evaluation": evaluation.id,
            "fold": fold,
            "age": age,
            "metrics": metrics,
        },
    )


def _decision(
    ledger: Ledger, *, kind: str, why: str, candidate: str, comparison: str | None
) -> str:
    decision_id = ledger.next_decision_id()
    ledger.put(
        Kinds.DECISION,
        decision_id,
        {
            "kind": kind,
            "why": why,
            "candidate": candidate,
            "comparison": comparison,
            "at": time.time(),
        },
    )
    return decision_id


def _set_baseline(ledger: Ledger, evaluation_id: str, candidate_id: str, decision: str):
    previous = ledger.get(Kinds.BASELINE, evaluation_id)
    if previous is not None and previous["candidate"] != candidate_id:
        ledger.update(
            Kinds.CANDIDATE,
            previous["candidate"],
            status=Status.SUPERSEDED,
            reason=f"superseded by {candidate_id[:8]} in decision {decision}",
        )
    row = {"candidate": candidate_id, "since": decision}
    if not ledger.put(Kinds.BASELINE, evaluation_id, row):
        ledger.update(Kinds.BASELINE, evaluation_id, **row)


def _latest_comparison(ledger: Ledger, candidate_id: str) -> str | None:
    rows = ledger.where(Kinds.COMPARISON, challenger=candidate_id)
    return rows[-1]["id"] if rows else None


def _candidate(ledger: Ledger, candidate_id: str) -> dict[str, Any]:
    cand = ledger.get(Kinds.CANDIDATE, candidate_id)
    if cand is None:
        raise KeyError(f"candidate {candidate_id} not found")
    return cand


# Metrics ==========================================================================================


def _aggregate_results(evaluation: Evaluation, results: list[Any]) -> dict[str, float]:
    from_series = evaluation.scorer.__forestry_meta__.get("from_series")
    if from_series is not None and all(r.series is not None for r in results):
        return from_series(np.concatenate([r.series for r in results]))
    names = results[0].metrics.keys()
    return {m: float(np.mean([r.metrics[m] for r in results])) for m in names}


def _fold_count(ledger: Ledger, candidate_id: str) -> int:
    rows = ledger.where(Kinds.SCORE, candidate=candidate_id)
    return len({s["fold"] for s in rows if s["fold"] is not None})


def _signed(a: dict[str, float], b: dict[str, float], m: str, directions: dict[str, str]) -> float:
    sign = 1.0 if directions[m] == "max" else -1.0
    return sign * (a[m] - b[m])


# Environment ======================================================================================


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
