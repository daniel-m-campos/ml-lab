"""The smallest ml-lab project: synthetic rows, one ridge pipeline, one evaluation.

Run it from the repository root:

    export ML_LAB_ROOT=$PWD/.ml-lab-minimal ML_LAB_ACTOR=$USER
    lab ingest examples/minimal/project.py
    lab run examples/minimal/project.py
    sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT name, metric, fold_mean FROM board"
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ml_lab import formats
from ml_lab.dataset import Dataset, Range, Segments, record
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Ledger
from ml_lab.splits import WalkForward


def dataset(ledger: Ledger, rows: str = "5000") -> str:
    """Rows of one feature and one noisy target, in a fixed order; prints the id."""
    random = np.random.default_rng(0)
    x = random.normal(size=int(rows))
    y = 0.5 * x + random.normal(size=int(rows))
    return record(
        ledger,
        Dataset({"x": x, "y": y}),
        source="toy",
        params={"rows": rows},
        targets=("y",),
    )


@dataclasses.dataclass(frozen=True)
class RidgeParams:
    alpha: float = 1.0


def ridge_fit(dataset: Dataset, train: Segments, params: RidgeParams) -> np.ndarray:
    x = dataset.column("x", train)
    y = dataset.column("y", train)
    return np.array([x @ y / (x @ x + params.alpha)])


def ridge_predict(model: np.ndarray, dataset: Dataset, rows: Range) -> np.ndarray:
    return model[0] * dataset.column("x", rows)


def ridge_save(model: np.ndarray) -> bytes:
    return formats.arrays_save({"slope": model})


def ridge_load(payload: bytes) -> np.ndarray:
    return formats.arrays_load(payload)["slope"]


def squared_error(
    pred: np.ndarray, dataset: Dataset, rows: Range, params
) -> np.ndarray:
    return (pred - dataset.column("y", rows)) ** 2


mse = Scorer(
    squared_error,
    metrics=lambda series: {"mse": float(series.mean())},
    directions={"mse": "min"},
)


ridge = Pipeline(
    name="ridge",
    fit=ridge_fit,
    predict=ridge_predict,
    save=ridge_save,
    load=ridge_load,
    format=formats.Format.ARROW_ARRAYS,
    params=RidgeParams(),
)
pipelines = [ridge, ridge.with_params(alpha=100.0).named("ridge_shrunk")]


def evaluations(dataset: str) -> list[Evaluation]:
    return [
        Evaluation(
            name="validation",
            dataset_id=dataset,
            split=WalkForward(first_cutoff_rows=2000, step_rows=1000, test_rows=1000),
            scorer=mse,
        )
    ]
