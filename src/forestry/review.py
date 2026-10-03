"""Read-only views over the log: the board, one pipeline in detail, the promotions.

Examples
--------
>>> board(ledger, evaluation)[0]["status"]  # doctest: +SKIP
'baseline'
"""

from __future__ import annotations

import json
from typing import Any

from forestry import harness
from forestry.declare import Evaluation
from forestry.ledger import Event, Ledger


def board(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """One row per pipeline's latest entry: status, verdict and deltas against the baseline."""
    rows = ledger.sql("SELECT * FROM board WHERE evaluation = ?", (evaluation.id,))
    age = str(evaluation.ages[0])
    out = []
    for row in rows:
        metrics = json.loads(row["aggregate"])[age]
        against = json.loads(row["baseline_aggregate"])[age] if row["baseline_aggregate"] else None
        compared = against is not None and row["entry"] != row["baseline"]
        out.append(
            {
                "entry": row["entry"],
                "id": row["pipeline"],
                "pipeline": row["name"],
                "status": row["status"],
                "verdict": harness.dominance(metrics, against, evaluation.directions)
                if compared
                else None,
                "deltas": harness.relative_deltas(metrics, against, evaluation.directions)
                if compared
                else {},
                "metrics": metrics,
                "why": _last_why(ledger, row["entry"]),
            }
        )
    out.sort(key=lambda r: r["status"] != "baseline")
    return out


def detail(ledger: Ledger, evaluation: Evaluation, pipeline_id: str) -> dict[str, Any]:
    """One pipeline: config against the baseline's, its entries with runs, folds, decisions."""
    rows = [r for r in board(ledger, evaluation) if r["id"] == pipeline_id]
    if not rows:
        raise KeyError(f"pipeline {pipeline_id} has no entry under this evaluation")
    row = rows[0]
    config = ledger.get(pipeline_id)["payload"]["config"]
    head = harness.baseline(ledger, evaluation)
    base_config = (
        ledger.get(ledger.get(head)["payload"]["pipeline"])["payload"]["config"] if head else None
    )
    entries = [
        e
        for e in ledger.events(Event.ENTRY, stream=evaluation.id)
        if e["payload"]["pipeline"] == pipeline_id
    ]
    return {
        **row,
        "config": config,
        "config_diff": _diff(config, base_config) if base_config else {},
        "entries": [
            {"entry": e["id"], "run": e["payload"]["run"], "at": e["at"], "actor": e["actor"]}
            for e in entries
        ],
        "folds": ledger.get(row["entry"])["payload"]["folds"],
        "decisions": [
            {
                "id": d["id"],
                "entry": d["key"],
                "kind": d["payload"]["kind"],
                "why": d["payload"]["why"],
            }
            for d in ledger.events(Event.DECISION, stream=evaluation.id)
            if d["payload"]["pipeline"] == pipeline_id
        ],
    }


def history(ledger: Ledger, evaluation: Evaluation) -> list[dict[str, Any]]:
    """Promotions under an evaluation, oldest first, with what each was made against."""
    return [
        {
            **d,
            "against": d["payload"].get("against"),
            "why": d["payload"]["why"],
            "pipeline": _name(ledger, d),
        }
        for d in ledger.events(Event.DECISION, stream=evaluation.id)
        if d["payload"]["kind"] == harness.Decision.PROMOTE
    ]


# Private Functions ================================================================================


def _name(ledger: Ledger, decision: dict[str, Any]) -> str:
    return ledger.get(decision["payload"]["pipeline"])["payload"]["name"]


def _last_why(ledger: Ledger, entry_id: str) -> str | None:
    decisions = ledger.events(Event.DECISION, key=entry_id)
    return decisions[-1]["payload"]["why"] if decisions else None


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {"this": a.get(k), "baseline": b.get(k)}
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }
