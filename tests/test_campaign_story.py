"""The campaign story as one test: freeze, declare, run, gate, compare, seal, deploy, refresh.

Mirrors docs/user-stories.md S1 to S9 on synthetic data; every object in docs/spec.md is touched.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from forestry import data, harness, review, toy
from forestry.declare import Evaluation, Pipeline, Schedule, Sealed, Stage, Tune, gates
from forestry.ledger import Ledger

FEATURES = ("f0", "f1", "f2")
TARGET = "ret_1"


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger.open(tmp_path / "forestry")


@pytest.fixture
def dataset(ledger: Ledger) -> str:
    return _freeze(ledger, months=12)


@pytest.fixture
def evaluation(dataset: str) -> Evaluation:
    return _evaluation(dataset)


@pytest.fixture
def ridge_1y() -> Pipeline:
    return Pipeline(
        name="ridge_1y",
        fit=toy.ridge_fit,
        predict=toy.ridge_predict,
        config=toy.RidgeConfig(features=FEATURES, target=TARGET, train_window_months=12, alpha=1.0),
    )


# S1 ===============================================================================================


def test_freezing_the_same_recipe_twice_returns_the_same_dataset_and_writes_nothing(ledger):
    first = _freeze(ledger, months=12)
    second = _freeze(ledger, months=12)
    assert first == second
    assert len(ledger.all("dataset")) == 1
    assert len(ledger.all("capture")) == 1


def test_a_dataset_records_its_recipe_hashes_and_rows(ledger, dataset):
    row = ledger.get("dataset", dataset)
    assert row["capture"] in {c["id"] for c in ledger.all("capture")}
    assert row["targets"] == [TARGET]
    assert row["rows"] > 0
    assert row["family"] == ["toy", "TOY"]


# S2 ===============================================================================================


def test_any_evaluation_field_change_is_a_new_evaluation_id(evaluation):
    changed_sim = _with_sim_cost(evaluation, cost=0.01)
    changed_schedule = dataclasses.replace(
        evaluation, schedule=dataclasses.replace(evaluation.schedule, ages=(1, 2))
    )
    assert len({evaluation.id, changed_sim.id, changed_schedule.id}) == 3


def test_the_schedule_expands_to_folds_with_embargo_and_no_sealed_rows(ledger, dataset, evaluation):
    session = data.session(ledger, dataset)
    folds, sealed = harness.expand(evaluation, session)
    assert len(folds) >= evaluation.min_folds
    embargo_rows = evaluation.schedule.embargo_seconds
    for fold in folds:
        train_end_ts = session.frame.ts[fold.train[1] - 1]
        eval_start_ts = session.frame.ts[fold.evals[1][0]]
        assert (eval_start_ts - train_end_ts).astype(int) >= embargo_rows
        for rng in fold.evals.values():
            assert rng[1] <= sealed[0]


# S3 ===============================================================================================


def test_running_a_pipeline_writes_one_fit_per_fold_and_reruns_compute_nothing(
    ledger, dataset, evaluation, ridge_1y
):
    session = data.session(ledger, dataset)
    folds, _ = harness.expand(evaluation, session)
    first = harness.run(ledger, [ridge_1y], evaluation)
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds) * len(evaluation.schedule.ages)
    second = harness.run(ledger, [ridge_1y], evaluation)
    assert second.fits_computed == 0
    assert second.predictions_computed == 0


def test_a_fit_row_carries_provenance(ledger, evaluation, ridge_1y):
    harness.run(ledger, [ridge_1y], evaluation)
    fit = ledger.all("fit")[0]
    for key in ("dataset", "pipeline", "train", "code_sha", "env_lock", "host", "duration_s"):
        assert key in fit, key
    assert fit["host"]["cpu_count"] >= 1


def test_heads_cost_one_fit_and_yield_one_candidate_each(ledger, dataset, evaluation):
    boost = Pipeline(
        name="boost",
        fit=toy.boost_fit,
        predict=toy.boost_predict,
        config=toy.BoostConfig(features=FEATURES, target=TARGET, n_iters=30, shrinkage=0.3),
        heads=(10, 20, 30),
    )
    session = data.session(ledger, dataset)
    folds, _ = harness.expand(evaluation, session)
    report = harness.run(ledger, [boost], evaluation)
    assert report.fits_computed == len(folds)
    assert report.predictions_computed == len(folds) * len(evaluation.schedule.ages) * 3
    heads = sorted(c["head"] for c in ledger.all("candidate"))
    assert heads == [10, 20, 30]


def test_a_tune_step_records_the_chosen_setting_per_fold(ledger, evaluation, ridge_1y):
    tuned = dataclasses.replace(
        ridge_1y,
        name="ridge_tuned",
        tune=Tune(space={"alpha": (0.1, 1.0, 10.0)}, inner_months=1, loss=toy.rmse_loss),
    )
    harness.run(ledger, [tuned], evaluation)
    chosen = [fit["artifacts"]["chosen"]["alpha"] for fit in ledger.all("fit")]
    assert chosen
    assert all(alpha in (0.1, 1.0, 10.0) for alpha in chosen)


# S4 and S5: stages, gates, comparison, review =====================================================


def test_stage_one_stops_at_the_human_gate_and_the_board_shows_it(ledger, evaluation, ridge_1y):
    ridge_3m = ridge_1y.with_config(train_window_months=3).named("ridge_3m")
    harness.run(ledger, [ridge_1y, ridge_3m], evaluation)
    board = review.board(ledger, evaluation)
    assert {row["pipeline"] for row in board} == {"ridge_1y", "ridge_3m"}
    assert all(row["stage"] == 0 and row["status"] == "pending" for row in board)
    assert all("corr" in row["metrics"] for row in board)


def test_a_human_gate_decision_is_recorded_and_a_stopped_candidate_keeps_its_reason(
    ledger, evaluation, ridge_1y
):
    ridge_3m = ridge_1y.with_config(train_window_months=3).named("ridge_3m")
    harness.run(ledger, [ridge_1y, ridge_3m], evaluation)
    good, bad = _candidates_by_pipeline(ledger, "ridge_1y", "ridge_3m")
    harness.gate(ledger, good, advance=True, why="corr holds across folds")
    harness.gate(ledger, bad, advance=False, why="corr negative after the drift")
    stopped = ledger.get("candidate", bad)
    assert stopped["status"] == "stopped"
    assert stopped["stopped_at"] == 0
    assert "drift" in stopped["reason"]
    kinds = [d["kind"] for d in ledger.all("decision")]
    assert kinds == ["gate", "gate"]


def test_the_funnel_fans_out_over_the_grid_prunes_to_the_front_and_seats_a_baseline(
    ledger, evaluation, ridge_1y
):
    harness.run(ledger, [ridge_1y], evaluation)
    (cand,) = _candidates_by_pipeline(ledger, "ridge_1y")
    harness.gate(ledger, cand, advance=True, why="ok")
    harness.run(ledger, [ridge_1y], evaluation)
    stage2 = [c for c in ledger.all("candidate") if c["exec"] is not None]
    assert len(stage2) == len(evaluation.stages[1].grid)
    survivors = [c for c in stage2 if c["stage"] >= 2]
    assert 1 <= len(survivors) <= evaluation.stages[1].gate.n
    baseline = review.baseline(ledger, evaluation)
    assert baseline is not None
    assert baseline["candidate"] in {c["id"] for c in survivors}
    assert ledger.all("comparison")
    assert any(d["kind"] == "promote" for d in ledger.all("decision"))


def test_a_baseline_moves_only_through_a_comparison(ledger, evaluation, ridge_1y):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    before = review.baseline(ledger, evaluation)["candidate"]
    ridge_6m = ridge_1y.with_config(train_window_months=6).named("ridge_6m")
    harness.run(ledger, [ridge_6m], evaluation)
    (cand,) = _candidates_by_pipeline(ledger, "ridge_6m")
    harness.gate(ledger, cand, advance=True, why="ok")
    harness.run(ledger, [ridge_6m], evaluation)
    comparisons = [c for c in ledger.all("comparison") if c["baseline"] == before]
    assert comparisons
    after = review.baseline(ledger, evaluation)["candidate"]
    moved = after != before
    verdicts = {c["verdict"] for c in comparisons}
    promoted = {d.get("candidate") for d in ledger.all("decision") if d["kind"] == "promote"}
    assert moved == ("dominates" in verdicts or after in promoted)
    if "incomparable" in verdicts and not moved:
        pending = [
            c for c in ledger.all("candidate") if c["status"] == "pending" and c["stage"] == 2
        ]
        assert pending
        harness.decide(ledger, kind="promote", candidate=pending[0]["id"], why="pnl matters more")
        assert review.baseline(ledger, evaluation)["candidate"] == pending[0]["id"]


def test_comparisons_are_refused_across_stages_and_evaluations(ledger, evaluation, ridge_1y):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    stage1 = next(c for c in ledger.all("candidate") if c["exec"] is None)
    with pytest.raises(harness.Refused):
        harness.compare(ledger, stage1["id"], evaluation)
    other = _with_sim_cost(evaluation, cost=0.01)
    final = review.baseline(ledger, evaluation)["candidate"]
    with pytest.raises(harness.Refused):
        harness.compare(ledger, final, other)
    assert review.history(ledger, other) == []


def test_review_answers_what_was_tried_why_it_lost_and_what_moved_the_baseline(
    ledger, evaluation, ridge_1y
):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    board = review.board(ledger, evaluation)
    assert all(
        {"pipeline", "head", "exec", "stage", "status", "metrics", "reason"} <= set(r)
        for r in board
    )
    history = review.history(ledger, evaluation)
    assert history and history[0]["kind"] == "promote"
    loser = next((c for c in ledger.all("candidate") if c["status"] == "stopped"), None)
    if loser is not None:
        why = review.why(ledger, loser["id"])
        assert why["reason"]
        assert why["folds"]
    found = review.find(ledger, pipeline="ridge_1y", train_window_months=12)
    assert found
    assert review.find(ledger, pipeline="ridge_1y", train_window_months=99) == []


# S6 and S9: seal, deploy, simulator change ========================================================


def test_the_sealed_window_is_scored_once_through_a_decision(ledger, evaluation, ridge_1y):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    session = data.session(ledger, evaluation.dataset)
    _, sealed = harness.expand(evaluation, session)
    assert not [p for p in ledger.all("predictions") if p["range"][0] >= sealed[0]]
    cand = review.baseline(ledger, evaluation)["candidate"]
    verdict = harness.seal(ledger, cand, evaluation)
    assert verdict.kind in ("seal-pass", "seal-fail")
    assert [p for p in ledger.all("predictions") if p["range"][0] >= sealed[0]]
    with pytest.raises(harness.Refused):
        harness.seal(ledger, cand, evaluation)


def test_deploy_requires_a_decision_and_the_bundle_names_its_justification(
    ledger, evaluation, ridge_1y
):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    cand = review.baseline(ledger, evaluation)["candidate"]
    with pytest.raises(harness.Refused):
        harness.deploy(ledger, cand, env="prod-toy")
    harness.decide(ledger, kind="deploy", candidate=cand, why="campaign done")
    deployment = harness.deploy(ledger, cand, env="prod-toy")
    bundle = json.loads(ledger.get_blob(deployment["bundle"]).decode())
    for key in ("dataset", "pipeline", "fit", "exec", "evaluation", "metrics", "code_sha", "host"):
        assert key in bundle, key
    assert ledger.get_blob(bundle["model"])


def test_a_simulator_change_leaves_old_comparisons_unconsulted(ledger, evaluation, ridge_1y):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    other = _with_sim_cost(evaluation, cost=0.01)
    assert review.baseline(ledger, other) is None
    harness.run(ledger, [ridge_1y], other)
    assert review.board(ledger, other)
    assert review.history(ledger, other) == []


# S7: refresh ======================================================================================


def test_a_refresh_seats_the_deployed_candidate_and_scores_it_first(ledger, evaluation, ridge_1y):
    _run_to_baseline(ledger, evaluation, ridge_1y)
    deployed = review.baseline(ledger, evaluation)["candidate"]
    harness.decide(ledger, kind="deploy", candidate=deployed, why="go")
    harness.deploy(ledger, deployed, env="prod-toy")

    refreshed = _freeze(ledger, months=15)
    assert refreshed != evaluation.dataset
    assert (
        ledger.get("dataset", refreshed)["family"]
        == ledger.get("dataset", evaluation.dataset)["family"]
    )
    new_eval = _evaluation(refreshed)
    seated = harness.seat(ledger, new_eval, deployed, why="incumbent from prod-toy")
    assert review.baseline(ledger, new_eval)["candidate"] == seated
    harness.run(ledger, [ridge_1y], new_eval)
    board = review.board(ledger, new_eval)
    assert board[0]["id"] == seated
    assert board[0]["stage"] == 2


# Helpers ==========================================================================================


def _freeze(ledger: Ledger, months: int) -> str:
    frame = toy.generate(
        start="2025-01-01", months=months, rows_per_day=40, seed=7, drift_at="2025-07-01"
    )
    capture = data.freeze_capture(
        ledger, frame, process="toy", params={"seed": 7, "rows_per_day": 40}, instrument="TOY"
    )
    return data.freeze_dataset(ledger, capture, filters=(), targets=(TARGET,))


def _evaluation(dataset: str) -> Evaluation:
    return Evaluation(
        dataset=dataset,
        schedule=Schedule(
            first_cutoff="2025-04-01",
            every_months=1,
            eval_months=1,
            ages=(1, 2, 3),
            embargo_seconds=3600,
        ),
        min_folds=3,
        sealed=Sealed(months=1, pass_rule="fold_range", fail="no_deploy"),
        stages=(
            Stage(scorer=toy.fit_metrics, gate=gates.human),
            Stage(
                scorer=toy.sign_sim,
                config=toy.SimConfig(cost=0.001, fidelity="quick"),
                grid=tuple(toy.Exec(threshold=t) for t in (0.0, 0.25, 0.5)),
                gate=gates.pareto_front(2),
            ),
            Stage(
                scorer=toy.sign_sim,
                config=toy.SimConfig(cost=0.001, fidelity="full"),
                gate=gates.vs_baseline,
            ),
        ),
        compare_age=1,
    )


def _with_sim_cost(evaluation: Evaluation, cost: float) -> Evaluation:
    stages = list(evaluation.stages)
    last = stages[-1]
    stages[-1] = dataclasses.replace(last, config=dataclasses.replace(last.config, cost=cost))
    return dataclasses.replace(evaluation, stages=tuple(stages))


def _candidates_by_pipeline(ledger: Ledger, *names: str) -> list[str]:
    by_name = {
        ledger.get("pipeline", c["pipeline"])["name"]: c["id"]
        for c in ledger.all("candidate")
        if c["exec"] is None
    }
    return [by_name[n] for n in names]


def _run_to_baseline(ledger: Ledger, evaluation: Evaluation, pipeline: Pipeline):
    harness.run(ledger, [pipeline], evaluation)
    (cand,) = _candidates_by_pipeline(ledger, pipeline.name)
    harness.gate(ledger, cand, advance=True, why="ok")
    harness.run(ledger, [pipeline], evaluation)
    assert review.baseline(ledger, evaluation) is not None
