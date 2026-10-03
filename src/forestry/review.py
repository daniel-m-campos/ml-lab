"""Read-only views over the ledger: what was tried, what moved the baseline, why one lost.

Examples
--------
>>> rows = board(ledger, evaluation)  # doctest: +SKIP
>>> rows[0]["status"]  # doctest: +SKIP
'advanced'
"""

from __future__ import annotations

import json
from typing import Any

from forestry import hashing
from forestry.declare import Evaluation
from forestry.ledger import Kinds, Ledger

# Public Functions =================================================================================


def baseline(ledger: Ledger, evaluation: Evaluation) -> dict[str, Any] | None:
    """The current baseline row for an evaluation, or None."""
    return ledger.get(Kinds.BASELINE, evaluation.id)


def board(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """One row per candidate: pipeline, head, exec, furthest stage, status, metrics, reason."""
    current = baseline(ledger, evaluation)
    top = current["candidate"] if current else None
    rows = [
        _board_row(ledger, evaluation, c)
        for c in ledger.where(Kinds.CANDIDATE, evaluation=evaluation.id)
    ]
    rows.sort(key=lambda r: (r["id"] != top, r["seq"]))
    return rows


def history(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """Promotions under an evaluation, oldest first, with their comparison verdicts."""
    out = []
    for decision in ledger.all(Kinds.DECISION):
        if decision["kind"] != "promote":
            continue
        cand = (
            ledger.get(Kinds.CANDIDATE, decision["candidate"])
            if decision.get("candidate")
            else None
        )
        if cand is None or cand["evaluation"] != evaluation.id:
            continue
        comparison = (
            ledger.get(Kinds.COMPARISON, decision["comparison"])
            if decision.get("comparison")
            else None
        )
        pipeline = ledger.get(Kinds.PIPELINE, cand["pipeline"])
        out.append(
            {
                **decision,
                "pipeline": pipeline["name"],
                "head": cand["head"],
                "exec": cand["exec"],
                "how": comparison["verdict"] if comparison else "seated",
            }
        )
    return out


def sealed(ledger: Ledger, candidate_id: str) -> dict[str, Any] | None:
    """The recorded seal verdict for a candidate: kind, metrics, decision id; None if unsealed."""
    for decision in ledger.all(Kinds.DECISION):
        if decision.get("candidate") == candidate_id and decision["kind"].startswith("seal"):
            return {
                "kind": decision["kind"],
                "metrics": json.loads(decision["why"])["sealed"],
                "decision": decision["id"],
            }
    return None


def why(ledger: Ledger, candidate_id: str) -> dict[str, Any]:
    """A candidate's config against the baseline's, its per-fold scores, what stopped it."""
    cand = ledger.get(Kinds.CANDIDATE, candidate_id)
    if cand is None:
        raise KeyError(candidate_id)
    pipeline = ledger.get(Kinds.PIPELINE, cand["pipeline"])
    current = ledger.get(Kinds.BASELINE, cand["evaluation"])
    base_pipeline = (
        ledger.get(Kinds.PIPELINE, ledger.get(Kinds.CANDIDATE, current["candidate"])["pipeline"])
        if current
        else None
    )
    folds = [
        {"fold": s["fold"], "age": s["age"], **s["metrics"]}
        for s in ledger.where(Kinds.SCORE, candidate=candidate_id, stage=cand["stage"])
        if s["fold"] is not None
    ]
    decisions = [d for d in ledger.all(Kinds.DECISION) if d.get("candidate") == candidate_id]
    return {
        "candidate": candidate_id,
        "pipeline": pipeline["name"],
        "config": pipeline["config"],
        "config_diff": _diff(pipeline["config"], base_pipeline["config"]) if base_pipeline else {},
        "exec": cand["exec"],
        "stage": cand["stage"],
        "status": cand["status"],
        "reason": cand.get("reason") or " ".join(d["why"] for d in decisions),
        "folds": folds,
    }


def find(ledger: Ledger, pipeline: str | None = None, **config: Any) -> list[dict[str, Any]]:
    """Candidates whose pipeline matches by name and config values."""
    out = []
    for cand in ledger.all(Kinds.CANDIDATE):
        row = ledger.get(Kinds.PIPELINE, cand["pipeline"])
        if pipeline is not None and row["name"] != pipeline:
            continue
        if any(row["config"].get(k) != hashing.canonical(v) for k, v in config.items()):
            continue
        out.append({**cand, "pipeline_name": row["name"]})
    return out


# Private Functions ================================================================================


def _board_row(ledger: Ledger, evaluation: Evaluation, cand: dict[str, Any]) -> dict[str, Any]:
    pipeline = ledger.get(Kinds.PIPELINE, cand["pipeline"])
    aggregate = [
        s
        for s in ledger.where(Kinds.SCORE, candidate=cand["id"], stage=cand["stage"], fold=None)
        if s["age"] == evaluation.compare_age
    ]
    seq = ledger.all(Kinds.CANDIDATE).index(cand)
    return {
        "id": cand["id"],
        "seq": seq,
        "pipeline": pipeline["name"],
        "head": cand["head"],
        "exec": cand["exec"],
        "stage": cand["stage"],
        "status": cand["status"],
        "metrics": aggregate[0]["metrics"] if aggregate else {},
        "reason": cand.get("reason"),
    }


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {"this": a.get(k), "baseline": b.get(k)}
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }
