"""The Optiver template on four stocks, driven through ``fy`` exactly as campaign.sh drives it.

Skipped when the Kaggle data is absent.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from forestry import harness, review  # noqa: E402
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


def test_the_script_steps_seat_an_incumbent_and_compare_the_rest(root, dataset):
    campaign = (DECL, "--dataset", dataset)
    fy(root, "run", *campaign)
    assert "scored" in fy(root, "board", *campaign)
    fy(root, "decide", *campaign, "ridge_3m", "--kind", "promote", "--why", "incumbent")
    compared = fy(root, "compare", *campaign)
    assert "pnl " in compared and "ridge_1m" in compared and "ridge_3m" not in compared
    assert "seated" in fy(root, "history", *campaign)
    assert '"config_diff"' in fy(root, "why", *campaign, "ridge_1m")


def test_a_rerun_computes_nothing(root, dataset):
    ledger = Ledger.open(root)
    evaluation = declarations.evaluation(dataset)
    report = harness.run(ledger, declarations.pipelines, evaluation)
    assert report.fits_computed == 0
    assert review.baseline(ledger, evaluation) is not None
