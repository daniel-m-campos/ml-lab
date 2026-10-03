"""The ledger story on synthetic data: freeze, declare, run, decide, read back.

Mirrors docs/user-stories.md; every object in docs/spec.md is written and read back.
"""

from __future__ import annotations

import dataclasses

import pytest

from forestry import data, harness, review
from forestry.ledger import Ledger
from tests import synthetic


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger.open(tmp_path / "forestry")


@pytest.fixture
def dataset(ledger: Ledger) -> str:
    return synthetic.freeze(ledger)


@pytest.fixture
def evaluation(dataset: str):
    return synthetic.evaluation(dataset)


def _seat(ledger, evaluation, pipeline) -> str:
    harness.run(ledger, [pipeline], evaluation)
    harness.decide(ledger, pipeline.id, evaluation, kind="promote", why="incumbent")
    return pipeline.id


# Dataset ==========================================================================================


def test_freezing_the_same_recipe_twice_returns_the_same_dataset_and_writes_nothing(ledger):
    assert synthetic.freeze(ledger) == synthetic.freeze(ledger)
    assert len(ledger.all("dataset")) == 1


def test_a_dataset_records_its_recipe_and_rows(ledger, dataset):
    row = ledger.get("dataset", dataset)
    assert (row["process"], row["instrument"]) == ("toy", "TOY")
    assert row["targets"] == [synthetic.TARGET]
    assert row["filters"][0]["__step__"].endswith("keep_all")
    assert row["rows"] > 0


# Declarations =====================================================================================


def test_a_pipeline_name_is_a_label_and_a_config_change_is_a_new_id():
    a = synthetic.ridge(3)
    assert a.named("other").id == a.id
    assert a.with_config(alpha=2.0).id != a.id


def test_any_evaluation_field_change_is_a_new_evaluation_id(dataset, evaluation):
    cost = synthetic.evaluation(dataset, cost=0.01)
    ages = dataclasses.replace(evaluation, ages=(1,))
    assert len({evaluation.id, cost.id, ages.id}) == 3


def test_the_schedule_expands_to_embargoed_folds(ledger, dataset, evaluation):
    session = data.session(ledger, dataset)
    folds = harness.expand(evaluation, session)
    assert len(folds) >= evaluation.min_folds
    for fold in folds:
        gap = session.ts[fold.evals[1][0]] - session.ts[fold.train[1] - 1]
        assert gap.astype(int) >= evaluation.embargo_seconds


# Run ==============================================================================================


def test_a_run_writes_one_fit_per_fold_and_a_rerun_computes_nothing(ledger, dataset, evaluation):
    folds = harness.expand(evaluation, data.session(ledger, dataset))
    first = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds) * len(evaluation.ages)
    second = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    assert (second.fits_computed, second.predictions_computed) == (0, 0)
    assert second.scored == first.scored


def test_a_scoring_change_refits_nothing(ledger, dataset, evaluation):
    harness.run(ledger, [synthetic.ridge(3)], evaluation)
    report = harness.run(ledger, [synthetic.ridge(3)], synthetic.evaluation(dataset, cost=0.01))
    assert report.fits_computed == 0
    assert len(ledger.all("score")) == 2


def test_a_fit_row_carries_provenance(ledger, evaluation):
    harness.run(ledger, [synthetic.ridge(3)], evaluation)
    fit = ledger.all("fit")[0]
    for key in ("dataset", "pipeline", "train", "cutoff", "code_sha", "env_lock", "host"):
        assert key in fit, key
    assert fit["host"]["cpu_count"] >= 1


def test_one_score_row_holds_every_fold_and_the_series_aggregate(ledger, dataset, evaluation):
    folds = harness.expand(evaluation, data.session(ledger, dataset))
    pipeline = synthetic.ridge(3)
    harness.run(ledger, [pipeline], evaluation)
    row = harness.score(ledger, pipeline.id, evaluation)
    assert len(row["folds"]) == len(folds) * len(evaluation.ages)
    assert set(row["aggregate"]) == {"1", "2"}
    summed = sum(f["pnl"] for f in row["folds"] if f["age"] == 1)
    assert harness.aggregate(ledger, pipeline.id, evaluation)["pnl"] == pytest.approx(summed)


# Decide ===========================================================================================


def test_decide_refuses_an_unscored_pipeline_and_promote_seats_a_baseline(ledger, evaluation):
    pipeline = synthetic.ridge(3)
    with pytest.raises(harness.Refused):
        harness.decide(ledger, pipeline.id, evaluation, kind="promote", why="x")
    _seat(ledger, evaluation, pipeline)
    assert harness.baseline(ledger, evaluation) == pipeline.id
    assert ledger.all("decision")[0]["against"] is None


def test_a_promotion_records_what_it_was_made_against(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    challenger = synthetic.ridge(1)
    harness.run(ledger, [challenger], evaluation)
    seen = harness.compare(ledger, challenger.id, evaluation)
    assert seen["baseline"] == base and set(seen["deltas"]) == set(evaluation.directions)
    harness.decide(ledger, challenger.id, evaluation, kind="promote", why="pnl up")
    assert ledger.all("decision")[-1]["against"] == seen
    assert harness.baseline(ledger, evaluation) == challenger.id


def test_reject_records_why_and_leaves_the_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    loser = synthetic.ridge(1)
    harness.run(ledger, [loser], evaluation)
    harness.decide(ledger, loser.id, evaluation, kind="reject", why="too much turnover")
    assert harness.baseline(ledger, evaluation) == base
    assert review.detail(ledger, evaluation, loser.id)["status"] == "rejected"


def test_a_pipeline_scored_under_another_evaluation_is_not_decidable_here(
    ledger, dataset, evaluation
):
    other = synthetic.evaluation(dataset, cost=0.01)
    harness.run(ledger, [synthetic.ridge(1)], other)
    with pytest.raises(harness.Refused):
        harness.decide(ledger, synthetic.ridge(1).id, evaluation, kind="promote", why="x")


def test_dominance_reads_directions():
    directions = {"pnl": "max", "turnover": "min"}
    better = harness.dominance({"pnl": 2, "turnover": 1}, {"pnl": 1, "turnover": 2}, directions)
    worse = harness.dominance({"pnl": 1, "turnover": 2}, {"pnl": 2, "turnover": 1}, directions)
    mixed = harness.dominance({"pnl": 2, "turnover": 2}, {"pnl": 1, "turnover": 1}, directions)
    assert (better, worse, mixed) == ("dominates", "dominated", "incomparable")


# Review ===========================================================================================


def test_the_board_answers_what_was_tried_and_what_moved_the_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    harness.run(ledger, synthetic.pipelines, evaluation)
    board = review.board(ledger, evaluation)
    assert board[0]["id"] == base and board[0]["status"] == "baseline"
    assert all(r["verdict"] and r["deltas"] for r in board[1:])
    assert {r["pipeline"] for r in board} == {"ridge_1m", "ridge_3m", "ridge_6m"}
    assert [h["against"] for h in review.history(ledger, evaluation)] == [None]
    detail = review.detail(ledger, evaluation, board[1]["id"])
    assert detail["config_diff"]["train_window_months"]["baseline"] == 6
    assert detail["folds"] and detail["status"] == "scored"
    harness.decide(ledger, board[1]["id"], evaluation, kind="promote", why="better")
    assert review.detail(ledger, evaluation, base)["status"] == "superseded"
