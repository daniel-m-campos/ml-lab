"""The ledger story on synthetic data: freeze, declare, run, compare, decide, review.

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


def _seat(ledger, evaluation, pipeline):
    report = harness.run(ledger, [pipeline], evaluation)
    harness.decide(ledger, report.candidates[0], kind="promote", why="incumbent")
    return report.candidates[0]


# Dataset ==========================================================================================


def test_freezing_the_same_recipe_twice_returns_the_same_dataset_and_writes_nothing(ledger):
    assert synthetic.freeze(ledger) == synthetic.freeze(ledger)
    assert len(ledger.all("dataset")) == 1
    assert len(ledger.all("capture")) == 1


def test_a_dataset_records_its_capture_filters_targets_and_rows(ledger, dataset):
    row = ledger.get("dataset", dataset)
    assert row["capture"] in {c["id"] for c in ledger.all("capture")}
    assert row["targets"] == [synthetic.TARGET]
    assert row["family"] == ["toy", "TOY"]
    assert row["rows"] > 0


# Declarations =====================================================================================


def test_a_pipeline_name_is_a_label_and_a_config_change_is_a_new_id():
    a = synthetic.ridge(3)
    assert a.named("other").id == a.id
    assert a.with_config(alpha=2.0).id != a.id


def test_any_evaluation_field_change_is_a_new_evaluation_id(dataset, evaluation):
    cost = synthetic.evaluation(dataset, cost=0.01)
    ages = dataclasses.replace(
        evaluation, schedule=dataclasses.replace(evaluation.schedule, ages=(1,))
    )
    assert len({evaluation.id, cost.id, ages.id}) == 3


def test_the_schedule_expands_to_embargoed_folds(ledger, dataset, evaluation):
    session = data.session(ledger, dataset)
    folds = harness.expand(evaluation, session)
    assert len(folds) >= evaluation.min_folds
    for fold in folds:
        train_end = session.frame.ts[fold.train[1] - 1]
        eval_start = session.frame.ts[fold.evals[1][0]]
        assert (eval_start - train_end).astype(int) >= evaluation.schedule.embargo_seconds


# Run ==============================================================================================


def test_a_run_writes_one_fit_per_fold_and_a_rerun_computes_nothing(ledger, dataset, evaluation):
    folds = harness.expand(evaluation, data.session(ledger, dataset))
    first = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds) * len(evaluation.schedule.ages)
    second = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    assert (second.fits_computed, second.predictions_computed) == (0, 0)
    assert second.candidates == first.candidates


def test_a_scoring_change_refits_nothing(ledger, dataset, evaluation):
    harness.run(ledger, [synthetic.ridge(3)], evaluation)
    report = harness.run(ledger, [synthetic.ridge(3)], synthetic.evaluation(dataset, cost=0.01))
    assert report.fits_computed == 0
    assert len(ledger.all("candidate")) == 2


def test_a_fit_row_carries_provenance(ledger, evaluation):
    harness.run(ledger, [synthetic.ridge(3)], evaluation)
    fit = ledger.all("fit")[0]
    for key in ("dataset", "pipeline", "train", "cutoff", "code_sha", "env_lock", "host"):
        assert key in fit, key
    assert fit["host"]["cpu_count"] >= 1


def test_scores_are_per_fold_and_age_with_a_series_aggregate(ledger, dataset, evaluation):
    folds = harness.expand(evaluation, data.session(ledger, dataset))
    report = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    rows = ledger.where("score", candidate=report.candidates[0])
    per_fold = [r for r in rows if r["fold"] is not None]
    aggregates = [r for r in rows if r["fold"] is None]
    assert len(per_fold) == len(folds) * len(evaluation.schedule.ages)
    assert {r["age"] for r in aggregates} == set(evaluation.schedule.ages)
    age_one = [r for r in per_fold if r["age"] == 1]
    summed = sum(r["metrics"]["pnl"] for r in age_one)
    assert harness.aggregate(ledger, report.candidates[0], 1)["pnl"] == pytest.approx(summed)


# Compare and decide ===============================================================================


def test_compare_refuses_without_a_baseline_and_promote_seats_one(ledger, evaluation):
    report = harness.run(ledger, [synthetic.ridge(3)], evaluation)
    with pytest.raises(harness.Refused):
        harness.compare(ledger, report.candidates[0], evaluation)
    harness.decide(ledger, report.candidates[0], kind="promote", why="incumbent")
    assert review.baseline(ledger, evaluation)["candidate"] == report.candidates[0]
    assert ledger.get("candidate", report.candidates[0])["status"] == "baseline"


def test_compare_writes_a_verdict_with_deltas_and_moves_nothing(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    challenger = harness.run(ledger, [synthetic.ridge(1)], evaluation).candidates[0]
    row = harness.compare(ledger, challenger, evaluation)
    assert row["verdict"] in ("dominates", "dominated", "incomparable")
    assert set(row["deltas"]) == set(evaluation.directions)
    assert review.baseline(ledger, evaluation)["candidate"] == base
    assert ledger.get("comparison", row["id"])["decision"] is None


def test_promote_links_the_comparison_and_supersedes_the_old_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    challenger = harness.run(ledger, [synthetic.ridge(1)], evaluation).candidates[0]
    comparison = harness.compare(ledger, challenger, evaluation)
    decision = harness.decide(ledger, challenger, kind="promote", why="pnl up, turnover fine")
    assert ledger.get("comparison", comparison["id"])["decision"] == decision
    assert review.baseline(ledger, evaluation)["candidate"] == challenger
    assert ledger.get("candidate", base)["status"] == "superseded"


def test_reject_records_why_and_leaves_the_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    loser = harness.run(ledger, [synthetic.ridge(1)], evaluation).candidates[0]
    harness.decide(ledger, loser, kind="reject", why="too much turnover")
    assert ledger.get("candidate", loser)["status"] == "rejected"
    assert review.baseline(ledger, evaluation)["candidate"] == base
    assert review.why(ledger, loser)["decisions"][-1]["why"] == "too much turnover"


def test_compare_refuses_a_candidate_from_another_evaluation(ledger, dataset, evaluation):
    _seat(ledger, evaluation, synthetic.ridge(6))
    other = synthetic.evaluation(dataset, cost=0.01)
    foreign = harness.run(ledger, [synthetic.ridge(1)], other).candidates[0]
    with pytest.raises(harness.Refused):
        harness.compare(ledger, foreign, evaluation)


def test_dominance_reads_directions():
    directions = {"pnl": "max", "turnover": "min"}
    assert harness.dominance({"pnl": 2, "turnover": 1}, {"pnl": 1, "turnover": 1}, directions)
    better = harness.dominance({"pnl": 2, "turnover": 1}, {"pnl": 1, "turnover": 2}, directions)
    worse = harness.dominance({"pnl": 1, "turnover": 2}, {"pnl": 2, "turnover": 1}, directions)
    mixed = harness.dominance({"pnl": 2, "turnover": 2}, {"pnl": 1, "turnover": 1}, directions)
    assert (better, worse, mixed) == ("dominates", "dominated", "incomparable")


# Review ===========================================================================================


def test_review_answers_what_was_tried_why_it_lost_and_what_moved_the_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    harness.run(ledger, synthetic.pipelines, evaluation)
    for cand in [c["id"] for c in ledger.all("candidate") if c["id"] != base]:
        harness.compare(ledger, cand, evaluation)
    board = review.board(ledger, evaluation)
    assert board[0]["id"] == base and board[0]["status"] == "baseline"
    assert all(r["verdict"] for r in board[1:])
    assert {r["pipeline"] for r in board} == {"ridge_1m", "ridge_3m", "ridge_6m"}
    history = review.history(ledger, evaluation)
    assert [h["verdict"] for h in history] == ["seated"]
    explained = review.why(ledger, board[1]["id"])
    assert explained["config_diff"]["train_window_months"]["baseline"] == 6
    assert explained["folds"] and explained["comparisons"]
    assert review.find(ledger, pipeline="ridge_1m", train_window_months=1)
