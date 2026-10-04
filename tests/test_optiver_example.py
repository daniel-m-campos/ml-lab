"""The Optiver template on four stocks, driven through ``fy`` exactly as campaign.sh drives it.

Skipped when the Kaggle data is absent.
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from forestry.ledger import Ledger  # noqa: E402
from optiver import capture, declarations  # noqa: E402

pytestmark = pytest.mark.skipif(not capture.available(), reason="Optiver train.csv not downloaded")
DECL = "optiver.declarations"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("optiver") / "forestry"


@pytest.fixture(scope="module")
def dataset(root: pathlib.Path) -> str:
    return fy(root, "ingest", "optiver.capture", "0,1,2,3").strip()


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


def test_the_dataset_is_parquet_with_a_null_free_target(root, dataset):
    ledger = Ledger(root)
    event = ledger.latest("dataset_recorded", dataset)
    table = pq.read_table(io.BytesIO(ledger.get_blob(event["payload"]["blob"]["sha"])))
    assert table.num_rows > 100_000 and table.num_rows == event["payload"]["rows"]
    assert not pc.any(pc.is_nan(table[capture.TARGET])).as_py()
    row = ledger.sql("SELECT process, instrument FROM dataset")[0]
    assert (row["process"], row["instrument"]) == (capture.PROCESS, capture.INSTRUMENT)


def test_the_script_steps_seat_an_incumbent_and_read_the_board(root, dataset):
    decl = (DECL, "--dataset", dataset)
    fy(root, "run", *decl)
    assert "scored" in fy(root, "board", *decl)
    fy(root, "decide", *decl, "ridge_3m", "--kind", "promote", "--why", "incumbent")
    board = fy(root, "board", *decl)
    assert "baseline" in board and "pnl " in board and "ridge_1m" in board
    assert "incumbent" in fy(root, "history", *decl)
    assert '"config_diff"' in fy(root, "board", *decl, "ridge_1m")


def test_a_rerun_writes_nothing_and_keeps_the_baseline(root, dataset):
    ledger = Ledger(root)
    before = len(ledger.events())
    out = fy(root, "run", DECL, "--dataset", dataset)
    assert out.startswith("up to date: 5 entries") or out.startswith("up to date: 3 entries")
    assert ledger.events()[before:] == []
    assert "baseline" in fy(root, "board", DECL, "--dataset", dataset)


@pytest.mark.skipif(not declarations.bonsai_available, reason="bonsai not installed")
def test_a_bonsai_model_blob_opens_with_bonsai_alone(root, dataset, tmp_path):
    import bonsai

    ledger = Ledger(root)
    fits = [
        f
        for f in ledger.events("fit_computed")
        if f["payload"]["model"]["format"] == "bonsai-msgpack"
    ]
    assert fits
    path = tmp_path / "model.msgpack"
    path.write_bytes(ledger.get_blob(fits[0]["payload"]["model"]["sha"]))
    assert bonsai.load(str(path)).n_iters > 0
