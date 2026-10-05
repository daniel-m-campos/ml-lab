"""The README's example, driven through ``lab`` as the README shows."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from ml_lab.ledger import Ledger

REPO = pathlib.Path(__file__).resolve().parents[1]
PROJECT = "examples/minimal/project.py"


def lab(root: pathlib.Path, *argv: str) -> str:
    env = {**os.environ, "ML_LAB_ROOT": str(root), "ML_LAB_ACTOR": "tester"}
    done = subprocess.run(
        [sys.executable, "-m", "ml_lab.cli", *argv],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return done.stdout


def test_the_minimal_project_ingests_runs_and_reads_back(tmp_path):
    root = tmp_path / "ml-lab"
    dataset = lab(root, "ingest", PROJECT).strip()
    first = lab(root, "run", PROJECT)
    assert "fits 6, predictions 6, scores 2" in first
    assert "nothing written" in lab(root, "run", PROJECT)
    board = Ledger(root).sql(
        "SELECT name, metric, fold_mean, n_folds FROM board ORDER BY fold_mean"
    )
    assert [r["name"] for r in board] == ["ridge", "ridge_shrunk"]
    assert board[0]["n_folds"] == 3 and 0.9 < board[0]["fold_mean"] < 1.1
    assert Ledger(root).sql("SELECT id FROM event_dataset")[0]["id"] == dataset
