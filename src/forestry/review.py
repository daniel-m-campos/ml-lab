"""Read-only views over the ledger: the board, one pipeline in detail, the promotions.

Examples
--------
>>> board(ledger, evaluation)[0]["status"]  # doctest: +SKIP
'baseline'
"""

from __future__ import annotations

from typing import Any

from forestry import harness
from forestry.declare import Evaluation
from forestry.ledger import Kinds, Ledger


class Status:
    SCORED = "scored"
    BASELINE = "baseline"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


def board(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """One row per scored pipeline: status, verdict and deltas against the baseline, metrics."""
    current = harness.baseline(ledger, evaluation)
    rows = [
        _row(ledger, evaluation, s["pipeline"], current)
        for s in ledger.where(Kinds.SCORE, evaluation=evaluation.id)
    ]
    rows.sort(key=lambda r: r["id"] != current)
    return rows


def detail(ledger: Ledger, evaluation: Evaluation, pipeline_id: str) -> dict[str, Any]:
    """One pipeline: config against the baseline's, per-fold scores, every decision on it."""
    row = _row(ledger, evaluation, pipeline_id, harness.baseline(ledger, evaluation))
    base = ledger.get(Kinds.PIPELINE, row["baseline"]) if row["baseline"] else None
    config = ledger.get(Kinds.PIPELINE, pipeline_id)["config"]
    return {
        **row,
        "config": config,
        "config_diff": _diff(config, base["config"]) if base else {},
        "folds": harness.score(ledger, pipeline_id, evaluation)["folds"],
        "decisions": _decisions(ledger, evaluation, pipeline_id),
    }


def history(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """Promotions under an evaluation, oldest first, with what each was made against."""
    return [
        {
            **d,
            "pipeline_id": d["pipeline"],
            "pipeline": ledger.get(Kinds.PIPELINE, d["pipeline"])["name"],
        }
        for d in ledger.where(Kinds.DECISION, evaluation=evaluation.id)
        if d["kind"] == harness.Decision.PROMOTE
    ]


# Private Functions ================================================================================


def _row(ledger: Ledger, evaluation: Evaluation, pipeline_id: str, current: str | None):
    decisions = _decisions(ledger, evaluation, pipeline_id)
    against = harness.compare(ledger, pipeline_id, evaluation) or {}
    return {
        "id": pipeline_id,
        "pipeline": ledger.get(Kinds.PIPELINE, pipeline_id)["name"],
        "status": _status(pipeline_id, current, decisions),
        "baseline": current,
        "verdict": against.get("verdict"),
        "deltas": against.get("deltas", {}),
        "metrics": harness.aggregate(ledger, pipeline_id, evaluation),
        "why": decisions[-1]["why"] if decisions else None,
    }


def _status(pipeline_id: str, current: str | None, decisions: list[dict[str, Any]]) -> str:
    if pipeline_id == current:
        return Status.BASELINE
    if not decisions:
        return Status.SCORED
    if decisions[-1]["kind"] == harness.Decision.REJECT:
        return Status.REJECTED
    return Status.SUPERSEDED


def _decisions(ledger: Ledger, evaluation: Evaluation, pipeline_id: str) -> list[dict[str, Any]]:
    return ledger.where(Kinds.DECISION, evaluation=evaluation.id, pipeline=pipeline_id)


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {"this": a.get(k), "baseline": b.get(k)}
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }
