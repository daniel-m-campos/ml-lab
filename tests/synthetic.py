"""Synthetic data, a ridge pipeline and a sign-trading scorer for the ledger story."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ml_lab import formats
from ml_lab.dataset import record
from ml_lab.dates import add_months, as_date
from ml_lab.experiment import Evaluation, Pipeline, scorer, step
from ml_lab.ledger import Ledger
from ml_lab.session import Range, Session
from ml_lab.splits import CalendarWalkForward, Segments

FEATURES = ("f0", "f1", "f2")
TARGET = "ret_1"
SECONDS_PER_DAY = 86_400


def generate(
    *, start: str, months: int, rows_per_day: int, seed: int, drift_at: str
) -> Session:
    """Three features and a linear target whose weights flip at ``drift_at``."""
    first = as_date(start)
    days = (add_months(first, months) - first).days
    rng = np.random.default_rng(seed)
    day_index = np.repeat(np.arange(days), rows_per_day)
    within = np.tile(
        np.linspace(0, SECONDS_PER_DAY - 1, rows_per_day, dtype=np.int64), days
    )
    ts = np.datetime64(first, "s") + (day_index * SECONDS_PER_DAY + within).astype(
        "timedelta64[s]"
    )
    features = rng.standard_normal((days * rows_per_day, len(FEATURES)))
    before, after = np.array([1.0, 0.5, -0.5]), np.array([-0.5, 1.0, 0.5])
    drift_row = int(np.searchsorted(ts, np.datetime64(as_date(drift_at), "s")))
    weights = np.where(np.arange(ts.shape[0])[:, None] < drift_row, before, after)
    target = 0.01 * np.sum(features * weights, axis=1) + 0.02 * rng.standard_normal(
        ts.shape[0]
    )
    columns = {name: features[:, i] for i, name in enumerate(FEATURES)}
    columns[TARGET] = target
    return Session(columns, ts)


@step
def keep_all(session: Session) -> Session:
    return session


def dataset(ledger: Ledger, months: int = 12, seed: int = 7) -> str:
    rows = generate(
        start="2025-01-01",
        months=months,
        rows_per_day=20,
        seed=seed,
        drift_at="2025-07-01",
    )
    return record(
        ledger,
        rows,
        source="synthetic",
        params={"months": months, "seed": seed, "instrument": "SYN"},
        filters=(keep_all,),
        targets=(TARGET,),
    )


@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    train_window_months: int
    alpha: float


@dataclasses.dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    bias: float


@step
def ridge_fit(session: Session, train: Segments, config: RidgeConfig) -> RidgeModel:
    rows = train
    if session.ts is not None and config.train_window_months:
        start, end = train[0][0], train[-1][1]
        since = session.index_of(
            add_months(session.date_at(end - 1), -config.train_window_months)
        )
        rows = ((max(start, since), end),)
    X = session.matrix(rows, FEATURES)
    y = session.column(TARGET, rows)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(
        Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean)
    )
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    return session.matrix(rng, FEATURES) @ model.weights + model.bias


FLAKY_CALLS: list[int] = []


@step
def flaky_fit(session: Session, train: Segments, config: RidgeConfig) -> RidgeModel:
    """Raises on its third call ever; the test clears ``FLAKY_CALLS`` to arm it."""
    FLAKY_CALLS.append(train[-1][1])
    if len(FLAKY_CALLS) == 3:
        raise RuntimeError("boom at the third fit")
    return ridge_fit(session, train, config)


@step(format=formats.Format.ARROW_ARRAYS)
def ridge_save(model: RidgeModel) -> bytes:
    return formats.arrays_save(
        {"weights": model.weights, "bias": np.array([model.bias])}
    )


@step
def ridge_load(payload: bytes) -> RidgeModel:
    arrays = formats.arrays_load(payload)
    return RidgeModel(arrays["weights"], float(arrays["bias"][0]))


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost: float


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    return {"pnl": float(series[:, 0].sum()), "turnover": float(series[:, 1].sum())}


@scorer(metrics=sim_metrics, directions={"pnl": "max", "turnover": "min"})
def sign_sim(
    pred: np.ndarray, session: Session, rng: Range, config: SimConfig
) -> np.ndarray:
    truth = session.column(TARGET, rng)
    position = np.sign(pred)
    flips = np.abs(np.diff(position, prepend=0.0))
    return np.column_stack([position * truth - config.cost * flips, flips])


def ridge(window_months: int, alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        name=f"ridge_{window_months}m",
        fit=ridge_fit,
        predict=ridge_predict,
        save=ridge_save,
        load=ridge_load,
        config=RidgeConfig(window_months, alpha),
    )


def evaluation(dataset: str, cost: float = 0.001, split: Any = None) -> Evaluation:
    return Evaluation(
        dataset=dataset,
        split=split
        or CalendarWalkForward(
            first_cutoff="2025-05-01", horizons=(1, 2), embargo_seconds=60, min_folds=3
        ),
        scorer=sign_sim,
        config=SimConfig(cost=cost),
    )


def flaky(window_months: int = 3) -> Pipeline:
    return dataclasses.replace(ridge(window_months), fit=flaky_fit, name="flaky")


pipelines = [ridge(1), ridge(3), ridge(6)]
