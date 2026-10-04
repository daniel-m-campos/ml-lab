"""Features, two model families and a per-stock taker simulation for Optiver."""

from __future__ import annotations

import dataclasses

import numpy as np

from forestry import formats
from forestry.dates import add_months
from forestry.experiment import scorer, step
from forestry.session import Range, Session
from forestry.splits import Segments

try:
    import bonsai
except ImportError:
    bonsai = None

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


# Features =============================================================================


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
            (col["bid_size"] - col["ask_size"])
            / (col["bid_size"] + col["ask_size"] + 1.0),
            (col["far_price"] - col["near_price"]) / wap * BPS,
            (col["near_price"] - wap) / wap * BPS,
            col["seconds_in_bucket"] / 540.0,
            np.log1p(col["matched_size"]),
        ]
    )
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def window(session: Session, train: Segments, months: int) -> Range:
    """The last ``months`` of the train segments as one range."""
    start, end = train[0][0], train[-1][1]
    return (
        max(start, session.index_of(add_months(session.date_at(end - 1), -months))),
        end,
    )


# Ridge ================================================================================


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
    rng = window(session, train, config.train_window_months)
    X = features(session, rng)
    y = session.column(TARGET, rng)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(
        Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean)
    )
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    return features(session, rng) @ model.weights + model.bias


@step(format=formats.Format.ARROW_ARRAYS)
def ridge_save(model: RidgeModel) -> bytes:
    return formats.arrays_save(
        {"weights": model.weights, "bias": np.array([model.bias])}
    )


@step
def ridge_load(payload: bytes) -> RidgeModel:
    arrays = formats.arrays_load(payload)
    return RidgeModel(arrays["weights"], float(arrays["bias"][0]))


# bonsai ===============================================================================


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
def bonsai_fit(session: Session, train: Segments, config: BonsaiConfig):
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
def bonsai_predict(model, session: Session, rng: Range) -> np.ndarray:
    return model.predict(features(session, rng))


@step(format="bonsai-msgpack")
def bonsai_save(model) -> bytes:
    return formats.bytes_via_file(model.save, ".msgpack")


@step
def bonsai_load(payload: bytes):
    return formats.load_via_file(payload, bonsai.BonsaiRegressor.from_file, ".msgpack")


# Scorers ==============================================================================


@dataclasses.dataclass(frozen=True)
class SimConfig:
    """Trade the sign of the predicted move above ``threshold_bps``; a flip costs
    ``cost_bps``.
    """

    cost_bps: float
    threshold_bps: float


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    """pnl, sharpe, max drawdown and turnover from a (pnl, flips) series."""
    pnl, flips = series[:, 0], series[:, 1]
    equity = np.cumsum(pnl)
    drawdown = (
        float(np.max(np.maximum.accumulate(equity) - equity)) if equity.size else 0.0
    )
    sharpe = (
        float(pnl.mean() / pnl.std() * np.sqrt(pnl.size))
        if pnl.size > 1 and pnl.std() > 0
        else 0.0
    )
    return {
        "pnl": float(pnl.sum()),
        "sharpe": sharpe,
        "max_dd": drawdown,
        "turnover": float(flips.sum()),
    }


@scorer(
    metrics=sim_metrics,
    directions={"pnl": "max", "sharpe": "max", "max_dd": "min", "turnover": "min"},
)
def taker_sim(
    pred: np.ndarray, session: Session, rng: Range, config: SimConfig
) -> np.ndarray:
    """Per-stock taker simulation in bps: a (pnl, flips) series."""
    truth = session.column(TARGET, rng)
    stock = session.column("stock_id", rng)
    position = np.sign(pred) * (np.abs(pred) > config.threshold_bps)
    flips = np.zeros_like(position)
    for sid in np.unique(stock):
        rows = np.flatnonzero(stock == sid)
        flips[rows] = np.abs(np.diff(position[rows], prepend=0.0))
    return np.column_stack([position * truth - config.cost_bps * flips, flips])
