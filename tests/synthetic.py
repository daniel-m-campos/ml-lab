"""Synthetic data, a ridge pipeline and a sign-trading scorer for the ledger story."""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
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
    columns: tuple[str, ...] = FEATURES


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
    X = session.matrix(rows, config.columns)
    y = session.column(TARGET, rows)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(
        Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean)
    )
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


@step
def ridge_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    columns = FEATURES + session.feature_columns
    return session.matrix(rng, columns) @ model.weights + model.bias


FEATURE_CALLS: list[int] = []


@step
def lagged_f0(session: Session) -> dict[str, np.ndarray]:
    """Yesterday's f0: row-causal, so the probe passes."""
    FEATURE_CALLS.append(session.rows)
    return {"f0_lag": np.concatenate([[np.nan], session.columns["f0"][:-1]])}


@step
def next_f0(session: Session) -> dict[str, np.ndarray]:
    """Tomorrow's f0: reads past the row, so the probe refuses it."""
    return {"f0_lag": np.roll(session.columns["f0"], -1)}


@step
def target_copy(session: Session) -> dict[str, np.ndarray]:
    return {"f0_lag": session.columns[TARGET]}


def featured(pipeline: Pipeline, features: Any = lagged_f0, name: str = "") -> Pipeline:
    config = dataclasses.replace(pipeline.config, columns=FEATURES + ("f0_lag",))
    return dataclasses.replace(
        pipeline, features=features, config=config, name=name or pipeline.name + "_lag"
    )


FLAKY_CALLS: list[int] = []
SEEN_ROWS: list[int] = []


@step
def peeking_fit(session: Session, train: Segments, config: RidgeConfig) -> RidgeModel:
    """Records how many rows it could see; the guard test reads it."""
    SEEN_ROWS.append(session.rows)
    return ridge_fit(session, train, config)


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


@step
def scale(pred: np.ndarray, session: Session, rng: Range, factor: float) -> np.ndarray:
    return pred * factor


@step(format=formats.Format.PICKLE)
def pickle_save(model: RidgeModel) -> bytes:
    return pickle.dumps(model)


@step
def pickle_load(payload: bytes) -> RidgeModel:
    return pickle.loads(payload)


@dataclasses.dataclass(frozen=True)
class BlendConfig:
    shrink: float = 0.0


@dataclasses.dataclass(frozen=True)
class BlendModel:
    weights: np.ndarray


@step
def blend_fit(
    session: Session, train: Segments, config: BlendConfig, members: list[np.ndarray]
) -> BlendModel:
    """Least squares on the members' in-sample predictions, shrunk toward equal."""
    x = np.column_stack(members)
    y = session.column(TARGET, train)
    fitted = np.linalg.lstsq(x, y, rcond=None)[0]
    equal = np.full(len(members), 1.0 / len(members))
    return BlendModel((1 - config.shrink) * fitted + config.shrink * equal)


@step
def blend_predict(
    model: BlendModel, session: Session, rng: Range, members: list[np.ndarray]
) -> np.ndarray:
    return np.column_stack(members) @ model.weights


@step(format=formats.Format.ARROW_ARRAYS)
def blend_save(model: BlendModel) -> bytes:
    return formats.arrays_save({"weights": model.weights})


@step
def blend_load(payload: bytes) -> BlendModel:
    return BlendModel(formats.arrays_load(payload)["weights"])


@step
def equal_fit(session: Session, train: Segments, config: BlendConfig) -> BlendModel:
    """Fixed equal weights: three arguments, so no in-sample member predictions."""
    return BlendModel(np.full(2, 0.5))


def blend(*members: Pipeline, shrink: float = 1.0) -> Pipeline:
    return Pipeline(
        name="blend_" + "_".join(m.name for m in members),
        fit=blend_fit,
        predict=blend_predict,
        save=blend_save,
        load=blend_load,
        config=BlendConfig(shrink),
        members=members,
    )


@step
def cheating_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    """Returns the target itself; the run hides it inside the window."""
    return session.column(TARGET, rng)


@step
def peeking_predict(model: RidgeModel, session: Session, rng: Range) -> np.ndarray:
    """Adds the window's mean feature to every row: reads the future inside it."""
    return ridge_predict(model, session, rng) + session.column("f0", rng).mean()


@step
def self_editing_fit(session: Session, train: Segments, config: RidgeConfig):
    """Appends a comment to its own module while fitting."""
    path = pathlib.Path(__file__)
    path.write_text(path.read_text() + "\n# edited during the run\n")
    return ridge_fit(session, train, config)


@step(format="tar")
def tar_save(model: RidgeModel) -> bytes:
    return b""


@step(format=formats.Format.ZIP)
def zip_save(model: RidgeModel) -> bytes:
    """A portable zip: the arrow-arrays model plus a text note."""
    return formats.zip_save(
        {
            "model.arrow": (formats.Format.ARROW_ARRAYS, ridge_save(model)),
            "note.txt": (formats.Format.TEXT, b"ridge"),
        }
    )


@step(format=formats.Format.ZIP)
def zip_pickle_save(model: RidgeModel) -> bytes:
    return formats.zip_save({"model.pkl": (formats.Format.PICKLE, pickle_save(model))})


@step
def zip_load(payload: bytes) -> RidgeModel:
    return ridge_load(formats.zip_load(payload)["model.arrow"])


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost: float


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    return {"pnl": float(series[:, 0].sum()), "turnover": float(series[:, 1].sum())}


@scorer(
    metrics=sim_metrics,
    directions={"pnl": "max", "turnover": "min"},
    columns=("pnl", "flips"),
)
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
            first_cutoff="2025-05-01",
            horizons=(1, 2),
            embargo_timestamps=1,
            min_folds=3,
        ),
        scorer=sign_sim,
        config=SimConfig(cost=cost),
    )


def flaky(window_months: int = 3) -> Pipeline:
    return dataclasses.replace(ridge(window_months), fit=flaky_fit, name="flaky")


pipelines = [ridge(1), ridge(3), ridge(6)]


def evaluations(dataset: str) -> list[Evaluation]:
    return [evaluation(dataset)]
