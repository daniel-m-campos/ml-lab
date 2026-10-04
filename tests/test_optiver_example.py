"""The Optiver template on four stocks, driven through ``lab`` exactly as launch.sh
drives it.

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

from ml_lab.ledger import Ledger  # noqa: E402
from optiver import dataset as optiver_dataset  # noqa: E402
from optiver import experiment  # noqa: E402

pytestmark = pytest.mark.skipif(
    not optiver_dataset.available(), reason="Optiver train.csv not downloaded"
)
DECL = "optiver.experiment"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("optiver") / "ml-lab"


@pytest.fixture(scope="module")
def dataset(root: pathlib.Path) -> str:
    return lab(root, "ingest", "optiver.dataset", "0,1,2,3").strip()


def lab(root: pathlib.Path, *argv: str) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "examples"),
        "ML_LAB_ROOT": str(root),
    }
    done = subprocess.run(
        [sys.executable, "-m", "ml_lab.cli", *argv],
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
    assert not pc.any(pc.is_nan(table[optiver_dataset.TARGET])).as_py()
    assert (
        ledger.sql("SELECT source FROM dataset")[0]["source"] == optiver_dataset.SOURCE
    )
    assert (
        event["payload"]["recipe"]["params"]["instrument"] == optiver_dataset.INSTRUMENT
    )


def test_the_run_scores_every_declared_pipeline_readable_by_sql(root, dataset):
    lab(root, "run", DECL)
    rows = Ledger(root).sql(
        "SELECT l.name, s.metric, s.value FROM latest_score l "
        "JOIN aggregate_score s ON s.score = l.score WHERE s.window = '1'"
    )
    names = {r["name"] for r in rows}
    assert {"ridge_1m", "ridge_3m", "ridge_6m"} <= names
    assert {r["metric"] for r in rows} == {"pnl", "sharpe", "max_dd", "turnover"}
    assert len(rows) == 4 * len(names)


def test_a_rerun_writes_nothing(root, dataset):
    ledger = Ledger(root)
    before = len(ledger.events())
    out = lab(root, "run", DECL, "--dataset", dataset)
    assert out.startswith("up to date:")
    assert ledger.events()[before:] == []


@pytest.mark.skipif(not experiment.bonsai_available, reason="bonsai not installed")
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
