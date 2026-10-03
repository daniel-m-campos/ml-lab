"""Decisions: promote or reject an entry, and the folds that read them back.

The baseline is the entry of the latest promote decision under an evaluation. ``decide`` records
what the decision was made against; nothing else moves the baseline.

Examples
--------
>>> decide(ledger, pipeline.id, evaluation, kind="promote", why="incumbent")  # doctest: +SKIP
>>> baseline(ledger, evaluation)  # doctest: +SKIP
'3f9c...'
"""

from __future__ import annotations

from typing import Any, Final

from forestry import hashing
from forestry.declare import Evaluation
from forestry.ledger import Event, Ledger, Refused


class Verdict:
    DOMINATES: Final = "dominates"
    DOMINATED: Final = "dominated"
    INCOMPARABLE: Final = "incomparable"


class Decision:
    PROMOTE: Final = "promote"
    REJECT: Final = "reject"


# Public Functions =================================================================================


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


# Private Functions ================================================================================


def _signed(a: dict[str, float], b: dict[str, float], m: str, directions: dict[str, str]) -> float:
    sign = 1.0 if directions[m] == "max" else -1.0
    return sign * (a[m] - b[m])
