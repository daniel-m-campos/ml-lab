"""Synthetic data, a ridge pipeline and two scorers for the ledger story."""

from __future__ import annotations

import dataclasses

import numpy as np

from forestry import data
from forestry.declare import Evaluation, Schedule, ScoreResult, scorer, step
from forestry.ledger import Ledger
from forestry.session import Frame, Range, Session, add_months, as_date

FEATURES = ("f0", "f1", "f2")
TARGET = "ret_1"
SECONDS_PER_DAY = 86_400


def generate(*, start: str, months: int, rows_per_day: int, seed: int, drift_at: str) -> Frame:
    """A time-ordered frame with three features and a linear target whose weights flip."""
    first = as_date(start)
    days = (add_months(first, months) - first).days
    rng = np.random.default_rng(seed)
    day_index = np.repeat(np.arange(days), rows_per_day)
    within = np.tile(np.linspace(0, SECONDS_PER_DAY - 1, rows_per_day, dtype=np.int64), days)
    ts = (
        np.datetime64(first, "s") + (day_index * SECONDS_PER_DAY + within).astype("timedelta64[s]")
    ).astype("datetime64[s]")
    features = rng.standard_normal((days * rows_per_day, len(FEATURES)))
    before, after = np.array([1.0, 0.5, -0.5]), np.array([-0.5, 1.0, 0.5])
    drift_row = int(np.searchsorted(ts, np.datetime64(as_date(drift_at), "s")))
    weights = np.where(np.arange(ts.shape[0])[:, None] < drift_row, before, after)
    target = 0.01 * np.sum(features * weights, axis=1) + 0.02 * rng.standard_normal(ts.shape[0])
    columns = {name: features[:, i] for i, name in enumerate(FEATURES)}
    columns[TARGET] = target
    return Frame(ts, columns)


@step
def keep_all(frame: Frame) -> Frame:
    return frame


def freeze(ledger: Ledger, months: int = 12, seed: int = 7) -> str:
    frame = generate(
        start="2025-01-01", months=months, rows_per_day=20, seed=seed, drift_at="2025-07-01"
    )
    capture = data.freeze_capture(
        ledger, frame, process="toy", params={"months": months, "seed": seed}, instrument="TOY"
    )
    return data.freeze_dataset(ledger, capture, filters=(keep_all,), targets=(TARGET,))


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
    end_date = session.frame.ts[train[1] - 1].astype("datetime64[D]").astype(object)
    start = max(train[0], session.index_of(add_months(end_date, -config.train_window_months)))
    X = session.matrix((start, train[1]), FEATURES)
    y = session.column(TARGET, (start, train[1]))
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    return session.matrix(rng, FEATURES) @ model.weights + model.bias


@scorer(directions={"corr": "max", "rmse": "min"})
def fit_metrics(pred: np.ndarray, session: Session, rng: Range, config: None) -> ScoreResult:
    truth = session.column(TARGET, rng)
    corr = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() > 0 else 0.0
    return ScoreResult({"corr": corr, "rmse": float(np.sqrt(np.mean((pred - truth) ** 2)))})


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost: float


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    return {"pnl": float(series[:, 0].sum()), "turnover": float(series[:, 1].sum())}


@scorer(directions={"pnl": "max", "turnover": "min"}, from_series=sim_metrics)
def sign_sim(pred: np.ndarray, session: Session, rng: Range, config: SimConfig) -> ScoreResult:
    truth = session.column(TARGET, rng)
    position = np.sign(pred)
    flips = np.abs(np.diff(position, prepend=0.0))
    series = np.column_stack([position * truth - config.cost * flips, flips])
    return ScoreResult(sim_metrics(series), series)


def ridge(window_months: int, alpha: float = 1.0):
    from forestry.declare import Pipeline

    return Pipeline(
        name=f"ridge_{window_months}m",
        fit=ridge_fit,
        predict=ridge_predict,
        config=RidgeConfig(window_months, alpha),
    )


def evaluation(dataset: str, cost: float = 0.001) -> Evaluation:
    return Evaluation(
        dataset=dataset,
        schedule=Schedule(first_cutoff="2025-05-01", ages=(1, 2), embargo_seconds=60),
        scorer=sign_sim,
        config=SimConfig(cost=cost),
        min_folds=3,
    )


pipelines = [ridge(1), ridge(3), ridge(6)]
