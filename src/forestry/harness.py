"""The harness: expands a schedule, memoizes fits and predictions, scores stage by stage, gates.

The harness owns ranges, the clock, the scorers, the gates and the sealed window. A pipeline
only sees ``fit(session, train_range, config)`` and ``predict(model, session, range, head)``.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> gate(ledger, report.candidates[0], advance=True, why="corr holds")  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import os
import pickle
import platform
import sys
import time
import uuid
from typing import Any, Final

import numpy as np

from forestry import data, hashing
from forestry.declare import Evaluation, Gate, Pipeline, Rule, Stage
from forestry.ledger import Kinds, Ledger
from forestry.session import Range, Session, add_months, as_date


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


class Status:
    PENDING: Final = "pending"
    ADVANCED: Final = "advanced"
    STOPPED: Final = "stopped"
    SEATED: Final = "seated"


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


@dataclasses.dataclass(frozen=True)
class SealVerdict:
    kind: str
    metrics: dict[str, float]
    decision: str


# Public Functions =================================================================================


def expand(evaluation: Evaluation, session: Session) -> tuple[list[Fold], Range]:
    """Folds from the schedule, never touching the sealed window; the sealed range second."""
    schedule = evaluation.schedule
    sealed_start = add_months(session.frame.end_exclusive, -evaluation.sealed.months)
    sealed = (session.index_of(sealed_start), session.frame.rows)
    span = max(schedule.ages) * schedule.eval_months
    folds: list[Fold] = []
    cutoff = as_date(schedule.first_cutoff)
    while add_months(cutoff, span) <= sealed_start:
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
    return folds, sealed


def run(ledger: Ledger, pipelines: list[Pipeline], evaluation: Evaluation) -> RunReport:
    """Fit, predict and score each pipeline under the evaluation, stopping at human gates."""
    _store_evaluation(ledger, evaluation)
    session = data.session(ledger, evaluation.dataset)
    folds, _ = expand(evaluation, session)
    report = RunReport()
    for pipeline in pipelines:
        _store_pipeline(ledger, pipeline)
        fits = [_ensure_fit(ledger, session, evaluation, pipeline, fold, report) for fold in folds]
        for fold, fit_id in zip(folds, fits, strict=True):
            for age, rng in fold.evals.items():
                for head in _heads(pipeline):
                    _ensure_predictions(
                        ledger, session, pipeline, fit_id, rng, head, fold.index, age, report
                    )
        _score_seated(ledger, session, evaluation, pipeline, folds)
        report.candidates += _advance_stages(ledger, session, evaluation, pipeline, folds)
    _settle_final_stage(ledger, evaluation)
    return report


def gate(ledger: Ledger, candidate_id: str, *, advance: bool, why: str) -> str:
    """Resolve a human gate with a recorded decision; returns the decision id."""
    cand = _candidate(ledger, candidate_id)
    if cand["status"] != Status.PENDING:
        raise Refused(f"candidate {candidate_id} is {cand['status']}, not pending")
    decision = _decision(ledger, kind="gate", why=why, candidate=candidate_id)
    if advance:
        ledger.update(Kinds.CANDIDATE, candidate_id, status=Status.ADVANCED, reason=why)
    else:
        ledger.update(
            Kinds.CANDIDATE,
            candidate_id,
            status=Status.STOPPED,
            stopped_at=cand["stage"],
            reason=why,
        )
    return decision


def compare(ledger: Ledger, candidate_id: str, evaluation: Evaluation) -> dict[str, Any]:
    """Pareto-compare a final-stage candidate against the baseline; writes the comparison row."""
    cand = _candidate(ledger, candidate_id)
    final_index = len(evaluation.stages) - 1
    if cand["evaluation"] != evaluation.id:
        raise Refused("candidate was scored under a different evaluation")
    if cand["stage"] != final_index:
        raise Refused(
            f"candidate is at stage {cand['stage']}, comparisons happen at stage {final_index}"
        )
    current = ledger.get(Kinds.BASELINE, evaluation.id)
    if current is None:
        raise Refused("no baseline under this evaluation")
    challenger = _aggregate(ledger, candidate_id, final_index, evaluation.compare_age)
    incumbent = _aggregate(ledger, current["candidate"], final_index, evaluation.compare_age)
    if _fold_count(ledger, candidate_id, final_index) < evaluation.min_folds:
        raise Refused("fewer folds than min_folds")
    verdict = _dominance(challenger, incumbent, evaluation.directions, evaluation.rule)
    row = {
        "challenger": candidate_id,
        "baseline": current["candidate"],
        "evaluation": evaluation.id,
        "verdict": verdict,
        "challenger_metrics": challenger,
        "baseline_metrics": incumbent,
        "decision": None,
    }
    comparison_id = uuid.uuid4().hex
    ledger.put(Kinds.COMPARISON, comparison_id, row)
    return {"id": comparison_id, **row}


def decide(
    ledger: Ledger,
    *,
    kind: str,
    why: str,
    candidate: str | None = None,
    comparison: str | None = None,
) -> str:
    """Record a human decision; promote moves the baseline, reject stops the candidate."""
    if comparison is None and candidate is not None:
        comparison = _latest_comparison(ledger, candidate)
    decision = _decision(ledger, kind=kind, why=why, candidate=candidate, comparison=comparison)
    if comparison is not None:
        ledger.update(Kinds.COMPARISON, comparison, decision=decision)
    if kind == "promote" and candidate is not None:
        cand = _candidate(ledger, candidate)
        _set_baseline(ledger, cand["evaluation"], cand["dataset"], candidate, decision)
        ledger.update(Kinds.CANDIDATE, candidate, status=Status.ADVANCED, reason=why)
    if kind == "reject" and candidate is not None:
        cand = _candidate(ledger, candidate)
        ledger.update(
            Kinds.CANDIDATE, candidate, status=Status.STOPPED, stopped_at=cand["stage"], reason=why
        )
    return decision


def seal(ledger: Ledger, candidate_id: str, evaluation: Evaluation) -> SealVerdict:
    """Score the sealed window once; pass when every metric sits inside the per-fold range."""
    cand = _candidate(ledger, candidate_id)
    if any(
        d["kind"].startswith("seal")
        for d in ledger.all(Kinds.DECISION)
        if d.get("candidate") == candidate_id
    ):
        raise Refused("the sealed window was already scored for this candidate")
    final_index = len(evaluation.stages) - 1
    if cand["stage"] != final_index:
        raise Refused("only final-stage candidates reach the seal")
    session = data.session(ledger, evaluation.dataset)
    _, sealed = expand(evaluation, session)
    pipeline = _load_pipeline(ledger, cand["pipeline"])
    sealed_start = session.frame.ts[sealed[0]].astype("datetime64[D]").astype(object)
    report = RunReport()
    fold = Fold(
        -1,
        sealed_start,
        (0, session.index_of(sealed_start, -evaluation.schedule.embargo_seconds)),
        {0: sealed},
    )
    fit_id = _ensure_fit(ledger, session, evaluation, pipeline, fold, report)
    pred = _ensure_predictions(
        ledger, session, pipeline, fit_id, sealed, cand["head"], -1, 0, report
    )
    stage = evaluation.final
    metrics = stage.scorer(pred, session, sealed, _load(cand["exec"]), stage.config).metrics
    bands = _fold_bands(ledger, candidate_id, final_index, evaluation.compare_age)
    passed = all(bands[m][0] <= v <= bands[m][1] for m, v in metrics.items() if m in bands)
    kind = "seal-pass" if passed else "seal-fail"
    decision = _decision(
        ledger,
        kind=kind,
        why=json.dumps({"sealed": metrics, "bands": bands}),
        candidate=candidate_id,
    )
    return SealVerdict(kind, metrics, decision)


def deploy(ledger: Ledger, candidate_id: str, *, env: str) -> dict[str, Any]:
    """Refit on the full window and record the bundle prod will load; needs a deploy decision."""
    cand = _candidate(ledger, candidate_id)
    decisions = [
        d
        for d in ledger.all(Kinds.DECISION)
        if d.get("candidate") == candidate_id and d["kind"] == "deploy"
    ]
    if not decisions:
        raise Refused("deploy needs a decision of kind deploy for this candidate")
    evaluation = _load_evaluation(ledger, cand["evaluation"])
    session = data.session(ledger, evaluation.dataset)
    pipeline = _load_pipeline(ledger, cand["pipeline"])
    fold = Fold(-2, session.frame.end_exclusive, (0, session.frame.rows), {})
    fit_id = _ensure_fit(ledger, session, evaluation, pipeline, fold, RunReport())
    fit = ledger.get(Kinds.FIT, fit_id)
    bundle = {
        "dataset": cand["dataset"],
        "pipeline": cand["pipeline"],
        "fit": fit_id,
        "model": fit["blob"],
        "head": cand["head"],
        "exec": cand["exec"],
        "evaluation": evaluation.id,
        "metrics": _aggregate(
            ledger, candidate_id, len(evaluation.stages) - 1, evaluation.compare_age
        ),
        "code_sha": fit["code_sha"],
        "env_lock": fit["env_lock"],
        "host": fit["host"],
    }
    bundle_sha = ledger.put_blob(json.dumps(bundle, sort_keys=True).encode())
    row = {
        "candidate": candidate_id,
        "bundle": bundle_sha,
        "env": env,
        "deployed_at": time.time(),
        "retired_at": None,
        "decision": decisions[-1]["id"],
    }
    deployment_id = uuid.uuid4().hex
    for other in ledger.where(Kinds.DEPLOYMENT, env=env, retired_at=None):
        ledger.update(Kinds.DEPLOYMENT, other["id"], retired_at=row["deployed_at"])
    ledger.put(Kinds.DEPLOYMENT, deployment_id, row)
    return {"id": deployment_id, **row}


def seat(ledger: Ledger, evaluation: Evaluation, candidate_id: str, *, why: str) -> str:
    """Seat a candidate from another dataset as this baseline; the next run scores it."""
    _store_evaluation(ledger, evaluation)
    source = _candidate(ledger, candidate_id)
    final_index = len(evaluation.stages) - 1
    seated_id = _ensure_candidate(
        ledger,
        evaluation,
        source["pipeline"],
        source["head"],
        source["exec"],
        parent=candidate_id,
        stage=final_index,
        status=Status.SEATED,
    )
    decision = _decision(ledger, kind="promote", why=why, candidate=seated_id)
    _set_baseline(ledger, evaluation.id, evaluation.dataset, seated_id, decision)
    return seated_id


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
    config, chosen = (
        _tune(session, evaluation, pipeline, fold) if pipeline.tune else (pipeline.config, {})
    )
    started = time.perf_counter()
    model = pipeline.fit(session, fold.train, config)
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
            "executor": "local",
            "duration_s": duration,
            "artifacts": {"chosen": chosen},
            "blob": ledger.put_blob(pickle.dumps(model)),
        },
    )
    report.fits_computed += 1
    return fit_id


def _tune(
    session: Session, evaluation: Evaluation, pipeline: Pipeline, fold: Fold
) -> tuple[Any, dict[str, Any]]:
    tune = pipeline.tune
    inner_cutoff = add_months(fold.cutoff, -tune.inner_months)
    embargo = evaluation.schedule.embargo_seconds
    inner_train = (fold.train[0], session.index_of(inner_cutoff, -embargo))
    validation = (session.index_of(inner_cutoff), fold.train[1])
    truth = session.column(_target_name(pipeline, evaluation, session), validation)
    best: tuple[float, dict[str, Any]] | None = None
    for values in itertools.product(*tune.space.values()):
        choice = dict(zip(tune.space.keys(), values, strict=True))
        model = pipeline.fit(session, inner_train, dataclasses.replace(pipeline.config, **choice))
        loss = tune.loss(pipeline.predict(model, session, validation, None), truth)
        if best is None or loss < best[0]:
            best = (loss, choice)
    chosen = best[1] if best else {}
    return dataclasses.replace(pipeline.config, **chosen), chosen


def _ensure_predictions(
    ledger: Ledger,
    session: Session,
    pipeline: Pipeline,
    fit_id: str,
    rng: Range,
    head: Any,
    fold: int,
    age: int,
    report: RunReport,
) -> np.ndarray:
    pred_id = hashing.content_hash({"fit": fit_id, "range": list(rng), "head": head})
    existing = ledger.get(Kinds.PREDICTIONS, pred_id)
    if existing is not None:
        return _load_array(ledger, existing["blob"])
    model = pickle.loads(ledger.get_blob(ledger.get(Kinds.FIT, fit_id)["blob"]))
    pred = np.asarray(pipeline.predict(model, session, rng, head), dtype=np.float64)
    ledger.put(
        Kinds.PREDICTIONS,
        pred_id,
        {
            "fit": fit_id,
            "range": list(rng),
            "head": head,
            "fold": fold,
            "age": age,
            "pipeline": pipeline.id,
            "blob": ledger.put_blob(pred.tobytes()),
        },
    )
    report.predictions_computed += 1
    return pred


# Stages ===========================================================================================


def _advance_stages(
    ledger: Ledger, session: Session, evaluation: Evaluation, pipeline: Pipeline, folds: list[Fold]
) -> list[str]:
    parents: list[dict[str, Any]] = []
    touched: list[str] = []
    for index, stage in enumerate(evaluation.stages):
        if index == 0:
            ids = [
                _ensure_candidate(ledger, evaluation, pipeline.id, head, None, parent=None, stage=0)
                for head in _heads(pipeline)
            ]
        elif stage.grid is not None:
            ids = [
                _ensure_candidate(
                    ledger, evaluation, pipeline.id, p["head"], exec, parent=p["id"], stage=index
                )
                for p in parents
                for exec in stage.grid
            ]
        else:
            ids = [p["id"] for p in parents]
            for cid in ids:
                ledger.update(Kinds.CANDIDATE, cid, stage=index, status=None)
        touched += ids
        for cid in ids:
            _score_candidate(
                ledger, session, evaluation, index, ledger.get(Kinds.CANDIDATE, cid), folds
            )
        candidates = [ledger.get(Kinds.CANDIDATE, cid) for cid in ids]
        _apply_gate(ledger, evaluation, index, stage, candidates)
        parents = [
            c
            for c in (ledger.get(Kinds.CANDIDATE, cid) for cid in ids)
            if c["stage"] == index and c["status"] == Status.ADVANCED
        ]
        if not parents:
            break
    return touched


def _score_candidate(
    ledger: Ledger,
    session: Session,
    evaluation: Evaluation,
    index: int,
    cand: dict[str, Any],
    folds: list[Fold],
):
    if ledger.where(Kinds.SCORE, candidate=cand["id"], stage=index, fold=None):
        return
    stage = evaluation.stages[index]
    exec = _load(cand["exec"])
    pipeline = _load_pipeline(ledger, cand["pipeline"])
    by_age: dict[int, list[Any]] = {}
    for fold in folds:
        fit_id = hashing.content_hash(
            {"dataset": evaluation.dataset, "pipeline": cand["pipeline"], "train": list(fold.train)}
        )
        for age, rng in fold.evals.items():
            pred = _ensure_predictions(
                ledger, session, pipeline, fit_id, rng, cand["head"], fold.index, age, RunReport()
            )
            result = stage.scorer(pred, session, rng, exec, stage.config)
            by_age.setdefault(age, []).append(result)
            ledger.put(
                Kinds.SCORE,
                hashing.content_hash(
                    {"candidate": cand["id"], "stage": index, "fold": fold.index, "age": age}
                ),
                {
                    "candidate": cand["id"],
                    "evaluation": evaluation.id,
                    "stage": index,
                    "fold": fold.index,
                    "age": age,
                    "metrics": result.metrics,
                },
            )
    for age, results in by_age.items():
        ledger.put(
            Kinds.SCORE,
            hashing.content_hash(
                {"candidate": cand["id"], "stage": index, "fold": None, "age": age}
            ),
            {
                "candidate": cand["id"],
                "evaluation": evaluation.id,
                "stage": index,
                "fold": None,
                "age": age,
                "metrics": _aggregate_results(stage, results),
            },
        )


def _apply_gate(
    ledger: Ledger,
    evaluation: Evaluation,
    index: int,
    stage: Stage,
    candidates: list[dict[str, Any]],
):
    fresh = [c for c in candidates if c["stage"] == index and c["status"] is None]
    if not fresh:
        return
    gate = stage.gate
    if gate.kind == "human":
        for c in fresh:
            ledger.update(Kinds.CANDIDATE, c["id"], status=Status.PENDING)
        return
    if gate.kind == "vs_baseline":
        if ledger.get(Kinds.BASELINE, evaluation.id) is None:
            return
        for c in fresh:
            _gate_vs_baseline(ledger, evaluation, c)
        return
    for parent in {c["parent"] for c in fresh}:
        group = [c for c in fresh if c["parent"] == parent]
        keep = _select(ledger, evaluation, index, gate, group)
        for c in group:
            if c["id"] in keep:
                ledger.update(
                    Kinds.CANDIDATE,
                    c["id"],
                    status=Status.ADVANCED,
                    reason=f"{gate.kind} at stage {index}",
                )
            else:
                ledger.update(
                    Kinds.CANDIDATE,
                    c["id"],
                    status=Status.STOPPED,
                    stopped_at=index,
                    reason=f"outside {gate.kind} at stage {index}",
                )


def _select(
    ledger: Ledger, evaluation: Evaluation, index: int, gate: Gate, group: list[dict[str, Any]]
) -> set[str]:
    directions = evaluation.stages[index].scorer.__forestry_meta__["directions"]
    metrics = {c["id"]: _aggregate(ledger, c["id"], index, evaluation.compare_age) for c in group}
    if gate.kind == "threshold":
        sign = 1 if directions[gate.metric] == "max" else -1
        return {cid for cid, m in metrics.items() if sign * m[gate.metric] >= sign * gate.minimum}
    if gate.kind == "top_k":
        reverse = directions[gate.metric] == "max"
        ranked = sorted(metrics, key=lambda cid: metrics[cid][gate.metric], reverse=reverse)
        return set(ranked[: gate.n])
    return set(_ranked(metrics, directions, evaluation.rule)[: gate.n])


def _settle_final_stage(ledger: Ledger, evaluation: Evaluation):
    """Seat the best of the first batch to reach a baseline gate, then gate the rest against it."""
    final = len(evaluation.stages) - 1
    if evaluation.stages[final].gate.kind != "vs_baseline":
        return
    fresh = [
        c
        for c in ledger.where(Kinds.CANDIDATE, evaluation=evaluation.id, stage=final)
        if c["status"] is None
    ]
    if not fresh:
        return
    if ledger.get(Kinds.BASELINE, evaluation.id) is None:
        _seat_best(ledger, evaluation, final, fresh)
    for c in fresh:
        _gate_vs_baseline(ledger, evaluation, c)


def _seat_best(ledger: Ledger, evaluation: Evaluation, index: int, group: list[dict[str, Any]]):
    metrics = {c["id"]: _aggregate(ledger, c["id"], index, evaluation.compare_age) for c in group}
    best = _ranked(metrics, evaluation.directions, evaluation.rule)[0]
    decision = _decision(
        ledger, kind="promote", why="best of the first batch at the final stage", candidate=best
    )
    _set_baseline(ledger, evaluation.id, evaluation.dataset, best, decision)
    ledger.update(Kinds.CANDIDATE, best, status=Status.ADVANCED, reason="first baseline")


def _gate_vs_baseline(ledger: Ledger, evaluation: Evaluation, cand: dict[str, Any]):
    current = ledger.get(Kinds.BASELINE, evaluation.id)
    if current["candidate"] == cand["id"]:
        ledger.update(Kinds.CANDIDATE, cand["id"], status=Status.ADVANCED)
        return
    comparison = compare(ledger, cand["id"], evaluation)
    if comparison["verdict"] == Verdict.DOMINATES:
        decision = _decision(
            ledger,
            kind="promote",
            why="dominates the baseline",
            candidate=cand["id"],
            comparison=comparison["id"],
        )
        _set_baseline(ledger, evaluation.id, evaluation.dataset, cand["id"], decision)
        ledger.update(Kinds.CANDIDATE, cand["id"], status=Status.ADVANCED, reason="dominates")
        ledger.update(Kinds.COMPARISON, comparison["id"], decision=decision)
    elif comparison["verdict"] == Verdict.DOMINATED:
        ledger.update(
            Kinds.CANDIDATE,
            cand["id"],
            status=Status.STOPPED,
            stopped_at=cand["stage"],
            reason="dominated by the baseline",
        )
    else:
        ledger.update(
            Kinds.CANDIDATE,
            cand["id"],
            status=Status.PENDING,
            reason="incomparable with the baseline, awaiting a decision",
        )


def _score_seated(
    ledger: Ledger, session: Session, evaluation: Evaluation, pipeline: Pipeline, folds: list[Fold]
):
    for cand in ledger.where(
        Kinds.CANDIDATE, evaluation=evaluation.id, pipeline=pipeline.id, status=Status.SEATED
    ):
        _score_candidate(ledger, session, evaluation, cand["stage"], cand, folds)
        ledger.update(Kinds.CANDIDATE, cand["id"], status=Status.ADVANCED)


# Rows =============================================================================================


def _ensure_candidate(
    ledger: Ledger,
    evaluation: Evaluation,
    pipeline_id: str,
    head: Any,
    exec: Any,
    *,
    parent: str | None,
    stage: int,
    status: str | None = None,
) -> str:
    key = {
        "dataset": evaluation.dataset,
        "pipeline": pipeline_id,
        "head": head,
        "exec": exec,
        "evaluation": evaluation.id,
    }
    cand_id = hashing.content_hash(key)
    ledger.put(
        Kinds.CANDIDATE,
        cand_id,
        {
            **key,
            "parent": parent,
            "stage": stage,
            "status": status,
            "stopped_at": None,
            "reason": None,
        },
    )
    return cand_id


def _decision(
    ledger: Ledger,
    *,
    kind: str,
    why: str,
    candidate: str | None = None,
    comparison: str | None = None,
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


def _set_baseline(
    ledger: Ledger, evaluation_id: str, dataset_id: str, candidate_id: str, decision_id: str
):
    family = ledger.get(Kinds.DATASET, dataset_id)["family"]
    body = {
        "evaluation": evaluation_id,
        "family": family,
        "candidate": candidate_id,
        "since": decision_id,
    }
    if not ledger.put(Kinds.BASELINE, evaluation_id, body):
        ledger.update(Kinds.BASELINE, evaluation_id, **body)


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


def _load_pipeline(ledger: Ledger, pipeline_id: str) -> Pipeline:
    return pickle.loads(ledger.get_blob(ledger.get(Kinds.PIPELINE, pipeline_id)["pickle"]))


def _load_evaluation(ledger: Ledger, evaluation_id: str) -> Evaluation:
    return pickle.loads(ledger.get_blob(ledger.get(Kinds.EVALUATION, evaluation_id)["pickle"]))


def _latest_comparison(ledger: Ledger, candidate_id: str) -> str | None:
    rows = ledger.where(Kinds.COMPARISON, challenger=candidate_id)
    return rows[-1]["id"] if rows else None


def _candidate(ledger: Ledger, candidate_id: str) -> dict[str, Any]:
    cand = ledger.get(Kinds.CANDIDATE, candidate_id)
    if cand is None:
        raise KeyError(f"candidate {candidate_id} not found")
    return cand


# Metrics ==========================================================================================


def _aggregate(ledger: Ledger, candidate_id: str, stage: int, age: int) -> dict[str, float]:
    rows = [
        s
        for s in ledger.where(Kinds.SCORE, candidate=candidate_id, stage=stage, fold=None)
        if s["age"] == age
    ]
    if not rows:
        raise Refused(f"candidate {candidate_id} has no aggregate score at stage {stage}")
    return rows[0]["metrics"]


def _aggregate_results(stage: Stage, results: list[Any]) -> dict[str, float]:
    from_series = stage.scorer.__forestry_meta__.get("from_series")
    if from_series is not None and all(r.series is not None for r in results):
        return from_series(np.concatenate([r.series for r in results]))
    names = results[0].metrics.keys()
    return {m: float(np.mean([r.metrics[m] for r in results])) for m in names}


def _fold_count(ledger: Ledger, candidate_id: str, stage: int) -> int:
    return len(
        {
            s["fold"]
            for s in ledger.where(Kinds.SCORE, candidate=candidate_id, stage=stage)
            if s["fold"] is not None
        }
    )


def _fold_bands(
    ledger: Ledger, candidate_id: str, stage: int, age: int
) -> dict[str, tuple[float, float]]:
    rows = [
        s
        for s in ledger.where(Kinds.SCORE, candidate=candidate_id, stage=stage)
        if s["fold"] is not None and s["age"] == age
    ]
    names = rows[0]["metrics"].keys()
    return {
        m: (min(r["metrics"][m] for r in rows), max(r["metrics"][m] for r in rows)) for m in names
    }


def _ranked(
    metrics: dict[str, dict[str, float]], directions: dict[str, str], rule: Rule
) -> list[str]:
    """Candidate ids best first: the priority order, or the Pareto front then the rest."""

    def cmp(x: str, y: str) -> int:
        verdict = _dominance(metrics[x], metrics[y], directions, rule)
        return -1 if verdict == Verdict.DOMINATES else 1 if verdict == Verdict.DOMINATED else 0

    def summed(cid: str) -> float:
        return -sum(metrics[cid][m] * (1 if d == "max" else -1) for m, d in directions.items())

    def wins(cid: str) -> int:
        return sum(cmp(cid, o) < 0 for o in metrics if o != cid)

    if rule.kind == "priority":
        return sorted(metrics, key=lambda cid: (-wins(cid), summed(cid)))
    front = [cid for cid in metrics if not any(cmp(o, cid) < 0 for o in metrics if o != cid)]
    rest = [cid for cid in metrics if cid not in front]
    return sorted(front, key=summed) + sorted(rest, key=summed)


def _dominance(
    a: dict[str, float], b: dict[str, float], directions: dict[str, str], rule: Rule
) -> str:
    def better(m: str) -> float:
        sign = 1.0 if directions[m] == "max" else -1.0
        return sign * (a[m] - b[m])

    if rule.kind == "priority":
        for m in rule.order:
            relative = better(m) / max(abs(b[m]), 1e-12)
            if relative > rule.band:
                return Verdict.DOMINATES
            if relative < -rule.band:
                return Verdict.DOMINATED
        return Verdict.INCOMPARABLE
    deltas = [better(m) for m in directions]
    if all(d >= 0 for d in deltas) and any(d > 0 for d in deltas):
        return Verdict.DOMINATES
    if all(d <= 0 for d in deltas) and any(d < 0 for d in deltas):
        return Verdict.DOMINATED
    return Verdict.INCOMPARABLE


# Environment ======================================================================================


def _heads(pipeline: Pipeline) -> tuple[Any, ...]:
    return pipeline.heads if pipeline.heads else (None,)


def _target_name(pipeline: Pipeline, evaluation: Evaluation, session: Session) -> str:
    return getattr(pipeline.config, "target", next(iter(session.frame.columns)))


def _load(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        return _Exec(**{k: v for k, v in value.items() if k != "__type__"})
    return value


class _Exec:
    """A canonical exec dict rehydrated as attribute access for scorers."""

    def __init__(self, **fields: Any):
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        return f"Exec({self.__dict__})"


def _load_array(ledger: Ledger, blob: str) -> np.ndarray:
    return np.frombuffer(ledger.get_blob(blob), dtype=np.float64)


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
