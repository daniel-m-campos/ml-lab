"""The smallest ml-lab project: synthetic rows, one ridge pipeline, one evaluation.

Run it from the repository root:

    export ML_LAB_ROOT=$PWD/.ml-lab-minimal
    lab ingest examples/minimal/project.py
    lab run examples/minimal/project.py
    sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT name, metric, fold_mean FROM board"
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ml_lab import formats
from ml_lab.dataset import record
from ml_lab.experiment import Evaluation, Pipeline, scorer, step
from ml_lab.ledger import Ledger
from ml_lab.session import Range, Session
from ml_lab.splits import Segments, WalkForward


def dataset(ledger: Ledger, rows: str = "5000") -> str:
    """Rows of one feature and one noisy target, in a fixed order; prints the id."""
    random = np.random.default_rng(0)
    x = random.normal(size=int(rows))
    y = 0.5 * x + random.normal(size=int(rows))
    session = Session({"x": x, "y": y})
    return record(
        ledger, session, source="toy", params={"rows": rows}, filters=(), targets=("y",)
    )


@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0


@step
def ridge_fit(session: Session, train: Segments, config: RidgeConfig) -> np.ndarray:
    x = session.column("x", train)
    y = session.column("y", train)
    return np.array([x @ y / (x @ x + config.alpha)])


@step
def ridge_predict(model: np.ndarray, session: Session, rng: Range) -> np.ndarray:
    return model[0] * session.column("x", rng)


@step(format=formats.Format.ARROW_ARRAYS)
def ridge_save(model: np.ndarray) -> bytes:
    return formats.arrays_save({"slope": model})


@step
def ridge_load(payload: bytes) -> np.ndarray:
    return formats.arrays_load(payload)["slope"]


@scorer(metrics=lambda series: {"mse": float(series.mean())}, directions={"mse": "min"})
def squared_error(pred: np.ndarray, session: Session, rng: Range, config) -> np.ndarray:
    return (pred - session.column("y", rng)) ** 2


ridge = Pipeline(
    name="ridge",
    fit=ridge_fit,
    predict=ridge_predict,
    save=ridge_save,
    load=ridge_load,
    config=RidgeConfig(),
)
pipelines = [ridge, ridge.with_config(alpha=100.0).named("ridge_shrunk")]


def evaluations(dataset: str) -> list[Evaluation]:
    return [
        Evaluation(
            name="validation",
            dataset=dataset,
            split=WalkForward(first_cutoff_rows=2000, step_rows=1000, window_rows=1000),
            scorer=squared_error,
        )
    ]
