"""Synthetic data, two toy pipelines, a loss, and two scorers for stories and tests.

The target is a linear signal whose weights flip at ``drift_at``, so a long training window
learns the wrong regime after the drift and walk-forward has something to show.

Examples
--------
>>> frame = generate(start="2025-01-01", months=2, rows_per_day=10, seed=1, drift_at="2025-02-01")
>>> frame.rows
590
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from forestry.declare import ScoreResult, scorer, step
from forestry.session import Frame, Range, Session, add_months, as_date

FEATURES = ("f0", "f1", "f2")
TARGET = "ret_1"
SECONDS_PER_DAY = 86_400


# Data =============================================================================================


def generate(*, start: str, months: int, rows_per_day: int, seed: int, drift_at: str) -> Frame:
    """A time-ordered frame with three features and a drifting linear target."""
    first = as_date(start)
    days = (add_months(first, months) - first).days
    rng = np.random.default_rng(seed)
    day_index = np.repeat(np.arange(days), rows_per_day)
    within = np.tile(np.linspace(0, SECONDS_PER_DAY - 1, rows_per_day, dtype=np.int64), days)
    ts = (
        np.datetime64(first, "s") + (day_index * SECONDS_PER_DAY + within).astype("timedelta64[s]")
    ).astype("datetime64[s]")
    features = rng.standard_normal((days * rows_per_day, len(FEATURES)))
    before = np.array([1.0, 0.5, -0.5])
    after = np.array([-0.5, 1.0, 0.5])
    drift_row = int(np.searchsorted(ts, np.datetime64(as_date(drift_at), "s")))
    weights = np.where(np.arange(ts.shape[0])[:, None] < drift_row, before, after)
    target = 0.01 * np.sum(features * weights, axis=1) + 0.02 * rng.standard_normal(ts.shape[0])
    columns = {name: features[:, i] for i, name in enumerate(FEATURES)}
    columns[TARGET] = target
    return Frame(ts, columns)


# Pipelines ========================================================================================


@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    """Ridge over a training window measured back from the cutoff."""

    features: tuple[str, ...]
    target: str
    train_window_months: int
    alpha: float


@dataclasses.dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    bias: float
    features: tuple[str, ...]


@step
def ridge_fit(session: Session, train: Range, config: RidgeConfig) -> RidgeModel:
    """Closed-form ridge on the last ``train_window_months`` of the training range."""
    window = _trim_to_window(session, train, config.train_window_months)
    X = session.matrix(window, config.features)
    y = session.column(config.target, window)
    weights, bias = _ridge(X, y, config.alpha)
    return RidgeModel(weights, bias, config.features)


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range, head: Any) -> np.ndarray:
    return session.matrix(rng, model.features) @ model.weights + model.bias


@dataclasses.dataclass(frozen=True)
class BoostConfig:
    """Shrunken ridge stages on residuals; a head is a stage count."""

    features: tuple[str, ...]
    target: str
    n_iters: int
    shrinkage: float


@dataclasses.dataclass(frozen=True)
class BoostModel:
    stages: tuple[tuple[np.ndarray, float], ...]
    shrinkage: float
    features: tuple[str, ...]


@step
def boost_fit(session: Session, train: Range, config: BoostConfig) -> BoostModel:
    X = session.matrix(train, config.features)
    residual = session.column(config.target, train).copy()
    stages = []
    for _ in range(config.n_iters):
        weights, bias = _ridge(X, residual, alpha=1.0)
        residual = residual - config.shrinkage * (X @ weights + bias)
        stages.append((weights, bias))
    return BoostModel(tuple(stages), config.shrinkage, config.features)


@step
def boost_predict(model: BoostModel, session: Session, rng: Range, head: Any) -> np.ndarray:
    X = session.matrix(rng, model.features)
    count = len(model.stages) if head is None else int(head)
    out = np.zeros(X.shape[0])
    for weights, bias in model.stages[:count]:
        out += model.shrinkage * (X @ weights + bias)
    return out


@step
def rmse_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# Scorers ==========================================================================================


@scorer(directions={"corr": "max", "hit_rate": "max", "rmse": "min"})
def fit_metrics(
    pred: np.ndarray, session: Session, rng: Range, exec: Any, config: Any
) -> ScoreResult:
    """Prediction quality against the target, no trading."""
    truth = session.column(TARGET, rng)
    corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 else 0.0
    hit = float(np.mean(np.sign(pred) == np.sign(truth)))
    return ScoreResult({"corr": corr, "hit_rate": hit, "rmse": rmse_loss(pred, truth)})


@dataclasses.dataclass(frozen=True)
class Exec:
    """Trade when the prediction exceeds ``threshold`` standard deviations."""

    threshold: float


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost: float
    fidelity: str


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    """pnl, sharpe, max drawdown and turnover from a (pnl, flips) series."""
    pnl, turnover = series[:, 0], series[:, 1]
    equity = np.cumsum(pnl)
    drawdown = float(np.max(np.maximum.accumulate(equity) - equity)) if equity.size else 0.0
    sharpe = (
        float(pnl.mean() / pnl.std() * np.sqrt(pnl.size)) if pnl.size > 1 and pnl.std() > 0 else 0.0
    )
    return {
        "pnl": float(pnl.sum()),
        "sharpe": sharpe,
        "max_dd": drawdown,
        "turnover": float(turnover.sum()),
    }


@scorer(
    directions={"pnl": "max", "sharpe": "max", "max_dd": "min", "turnover": "min"},
    from_series=sim_metrics,
)
def sign_sim(
    pred: np.ndarray, session: Session, rng: Range, exec: Exec, config: SimConfig
) -> ScoreResult:
    """Taker simulation: hold the sign of the prediction above a threshold, pay a cost per flip."""
    truth = session.column(TARGET, rng)
    scale = pred.std() if pred.std() > 0 else 1.0
    position = np.sign(pred) * (np.abs(pred) > exec.threshold * scale)
    flips = np.abs(np.diff(position, prepend=0.0))
    pnl = position * truth - config.cost * flips
    series = np.column_stack([pnl, flips])
    return ScoreResult(sim_metrics(series), series)


# Private Functions ================================================================================


def _trim_to_window(session: Session, train: Range, months: int) -> Range:
    end_date = session.frame.ts[train[1] - 1].astype("datetime64[D]").astype(object)
    start = max(train[0], session.index_of(add_months(end_date, -months)))
    return (start, train[1])


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    weights = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return weights, float(y_mean - x_mean @ weights)
