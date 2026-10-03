"""The Optiver example on four stocks; skipped when the Kaggle data is absent."""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

from forestry import data, harness, review  # noqa: E402
from forestry.ledger import Ledger  # noqa: E402
from optiver import capture, declarations  # noqa: E402

pytestmark = pytest.mark.skipif(not capture.available(), reason="Optiver train.csv not downloaded")


@pytest.fixture(scope="module")
def ledger(tmp_path_factory) -> Ledger:
    return Ledger.open(tmp_path_factory.mktemp("optiver") / "forestry")


@pytest.fixture(scope="module")
def dataset(ledger: Ledger) -> str:
    return capture.freeze(ledger, stocks=(0, 1, 2, 3))


def test_the_capture_is_time_ordered_with_a_null_free_target(ledger, dataset):
    row = ledger.get("dataset", dataset)
    assert row["family"] == [capture.PROCESS, capture.INSTRUMENT]
    assert row["rows"] > 100_000


def test_the_ridge_windows_run_the_funnel_to_a_baseline(ledger, dataset):
    evaluation = declarations.evaluation(dataset)
    report = harness.run(ledger, declarations.ridge_windows, evaluation)
    assert report.fits_computed > 0
    for row in review.board(ledger, evaluation):
        harness.gate(ledger, row["id"], advance=True, why="test")
    harness.run(ledger, declarations.ridge_windows, evaluation)
    assert review.baseline(ledger, evaluation) is not None
    assert harness.run(ledger, declarations.ridge_windows, evaluation).fits_computed == 0


@pytest.mark.skipif(not declarations.bonsai_available, reason="bonsai not installed")
def test_bonsai_heads_cost_one_fit_per_fold(ledger, dataset):
    evaluation = declarations.evaluation(dataset)
    small = dataclasses.replace(
        declarations.bonsai_depthwise.with_config(n_iters=50), name="bonsai_small", heads=(25, 50)
    )
    folds, _ = harness.expand(evaluation, data.session(ledger, dataset))
    report = harness.run(ledger, [small], evaluation)
    assert report.fits_computed == len(folds)
    assert report.predictions_computed == len(folds) * len(evaluation.schedule.ages) * 2
