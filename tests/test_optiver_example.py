"""The Optiver template on four stocks, driven through ``fy`` exactly as campaign.sh drives it.

Skipped when the Kaggle data is absent.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from forestry import data, harness, review  # noqa: E402
from forestry.ledger import Ledger  # noqa: E402
from optiver import capture, declarations  # noqa: E402

pytestmark = pytest.mark.skipif(not capture.available(), reason="Optiver train.csv not downloaded")
DECL = "optiver.declarations"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("optiver") / "forestry"


@pytest.fixture(scope="module")
def dataset(root: pathlib.Path) -> str:
    return fy(root, "freeze", "optiver.capture:freeze", "0,1,2,3").strip()


def fy(root: pathlib.Path, *argv: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO / "examples"), "FORESTRY_ROOT": str(root)}
    done = subprocess.run(
        [sys.executable, "-m", "forestry.cli", *argv],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return done.stdout


def test_the_capture_is_time_ordered_with_a_null_free_target(root, dataset):
    row = Ledger.open(root).get("dataset", dataset)
    assert row["family"] == [capture.PROCESS, capture.INSTRUMENT]
    assert row["rows"] > 100_000


def test_the_script_steps_reach_a_sealed_baseline(root, dataset):
    fy(root, "run", DECL, "--dataset", dataset)
    board = fy(root, "board", DECL, "--dataset", dataset)
    assert "pending" in board and "corr=" in board
    fy(root, "gate", DECL, "--dataset", dataset, "--pending", "--advance", "--why", "test")
    fy(root, "run", DECL, "--dataset", dataset)
    assert "seated" in fy(root, "history", DECL, "--dataset", dataset)
    assert fy(root, "seal", DECL, "--dataset", dataset).startswith("seal")


def test_a_rerun_computes_nothing(root, dataset):
    ledger = Ledger.open(root)
    report = harness.run(ledger, declarations.pipelines, declarations.evaluation(dataset))
    assert report.fits_computed == 0
    assert review.baseline(ledger, declarations.evaluation(dataset)) is not None


@pytest.mark.skipif(not declarations.bonsai_available, reason="bonsai not installed")
def test_bonsai_heads_cost_one_fit_per_fold(root, dataset):
    ledger = Ledger.open(root)
    evaluation = declarations.evaluation(dataset)
    small = dataclasses.replace(
        declarations.bonsai_depthwise.with_config(n_iters=50), name="bonsai_small", heads=(25, 50)
    )
    folds, _ = harness.expand(evaluation, data.session(ledger, dataset))
    report = harness.run(ledger, [small], evaluation)
    assert report.fits_computed == len(folds)
    assert report.predictions_computed == len(folds) * len(evaluation.schedule.ages) * 2
