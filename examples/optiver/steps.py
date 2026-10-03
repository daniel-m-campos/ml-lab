"""Features, two model families and a per-stock taker simulation for the Optiver data."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from forestry.declare import ScoreResult, scorer, step
from forestry.session import Range, Session, add_months
from forestry.toy import sim_metrics

TARGET = "target"
BPS = 1e4
BOOK_COLUMNS = (
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "matched_size",
    "reference_price",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
    "seconds_in_bucket",
)


# Features =========================================================================================


def features(session: Session, rng: Range) -> np.ndarray:
    """Public-notebook style features over a range; NaN becomes 0."""
    col = {name: session.column(name, rng) for name in BOOK_COLUMNS}
    wap = col["wap"]
    X = np.column_stack(
        [
            col["imbalance_size"] / (col["matched_size"] + 1.0),
            col["imbalance_buy_sell_flag"],
            (col["ask_price"] - col["bid_price"]) / wap * BPS,
            (wap - col["reference_price"]) / wap * BPS,
            (col["bid_size"] - col["ask_size"]) / (col["bid_size"] + col["ask_size"] + 1.0),
            (col["far_price"] - col["near_price"]) / wap * BPS,
            (col["near_price"] - wap) / wap * BPS,
            col["seconds_in_bucket"] / 540.0,
            np.log1p(col["matched_size"]),
        ]
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def window(session: Session, train: Range, months: int) -> Range:
    end_date = session.frame.ts[train[1] - 1].astype("datetime64[D]").astype(object)
    return (max(train[0], session.index_of(add_months(end_date, -months))), train[1])


# Ridge ============================================================================================


@dataclasses.dataclass(frozen=True)
class RidgeConfig:
    train_window_months: int
    alpha: float


@dataclasses.dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    bias: float


@step
def ridge_fit(session: Session, train: Range, config: RidgeConfig) -> RidgeModel:
    rng = window(session, train, config.train_window_months)
    X = features(session, rng)
    y = session.column(TARGET, rng)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range, head: Any) -> np.ndarray:
    return features(session, rng) @ model.weights + model.bias


# bonsai ===========================================================================================


@dataclasses.dataclass(frozen=True)
class BonsaiConfig:
    train_window_months: int
    grower: str
    max_depth: int
    n_iters: int
    learning_rate: float
    max_bin: int
    n_threads: int = 0


@step
def bonsai_fit(session: Session, train: Range, config: BonsaiConfig):
    import bonsai

    rng = window(session, train, config.train_window_months)
    model = bonsai.BonsaiRegressor(
        n_iters=config.n_iters,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        grower=config.grower,
        max_bin=config.max_bin,
        n_threads=config.n_threads,
    )
    return model.fit(features(session, rng), session.column(TARGET, rng))


@step
def bonsai_predict(model, session: Session, rng: Range, head: Any) -> np.ndarray:
    return model.predict(features(session, rng), num_iteration=0 if head is None else int(head))


# Scorers ==========================================================================================


@scorer(directions={"corr": "max", "hit_rate": "max", "rmse": "min"})
def fit_metrics(
    pred: np.ndarray, session: Session, rng: Range, exec: Any, config: Any
) -> ScoreResult:
    truth = session.column(TARGET, rng)
    corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 else 0.0
    return ScoreResult(
        {
            "corr": corr,
            "hit_rate": float(np.mean(np.sign(pred) == np.sign(truth))),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        }
    )


@dataclasses.dataclass(frozen=True)
class Exec:
    """Trade the sign of the predicted move when it exceeds ``threshold_bps``."""

    threshold_bps: float


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost_bps: float
    fidelity: str


@scorer(
    directions={"pnl": "max", "sharpe": "max", "max_dd": "min", "turnover": "min"},
    from_series=sim_metrics,
)
def taker_sim(
    pred: np.ndarray, session: Session, rng: Range, exec: Exec, config: SimConfig
) -> ScoreResult:
    """Per-stock taker simulation in bps: position flips cost ``cost_bps`` each."""
    truth = session.column(TARGET, rng)
    stock = session.column("stock_id", rng)
    position = np.sign(pred) * (np.abs(pred) > exec.threshold_bps)
    flips = np.zeros_like(position)
    for sid in np.unique(stock):
        rows = np.flatnonzero(stock == sid)
        flips[rows] = np.abs(np.diff(position[rows], prepend=0.0))
    pnl = position * truth - config.cost_bps * flips
    series = np.column_stack([pnl, flips])
    return ScoreResult(sim_metrics(series), series)
