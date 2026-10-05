"""Synthetic data, a ridge pipeline and a sign-trading scorer for the ledger story."""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
from typing import Any

import numpy as np

from ml_lab import formats
from ml_lab.dataset import Dataset, Range, Segments, record
from ml_lab.dates import add_months, as_date
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Ledger
from ml_lab.splits import CalendarWalkForward

FEATURES = ("f0", "f1", "f2")
TARGET = "ret_1"
SECONDS_PER_DAY = 86_400


def generate(
    *, start: str, months: int, rows_per_day: int, seed: int, drift_at: str
) -> Dataset:
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
    return Dataset(columns, ts)


def keep_all(dataset: Dataset) -> Dataset:
    return dataset


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


def ridge_fit(dataset: Dataset, train: Segments, config: RidgeConfig) -> RidgeModel:
    rows = train
    if dataset.ts is not None and config.train_window_months:
        start, end = train[0][0], train[-1][1]
        since = dataset.index_of(
            add_months(dataset.date_at(end - 1), -config.train_window_months)
        )
        rows = ((max(start, since), end),)
    X = dataset.matrix(rows, config.columns)
    y = dataset.column(TARGET, rows)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(
        Xc.T @ Xc + config.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean)
    )
    return RidgeModel(weights, float(y_mean - x_mean @ weights))


def ridge_predict(model: RidgeModel, dataset: Dataset, rng: Range) -> np.ndarray:
    columns = FEATURES + dataset.feature_columns
    return dataset.matrix(rng, columns) @ model.weights + model.bias


FEATURE_CALLS: list[int] = []


def lagged_f0(dataset: Dataset) -> dict[str, np.ndarray]:
    """Yesterday's f0: row-causal, so the probe passes."""
    FEATURE_CALLS.append(dataset.rows)
    return {"f0_lag": np.concatenate([[np.nan], dataset.columns["f0"][:-1]])}


def next_f0(dataset: Dataset) -> dict[str, np.ndarray]:
    """Tomorrow's f0: reads past the row, so the probe refuses it."""
    return {"f0_lag": np.roll(dataset.columns["f0"], -1)}


def target_copy(dataset: Dataset) -> dict[str, np.ndarray]:
    return {"f0_lag": dataset.columns[TARGET]}


def featured(pipeline: Pipeline, features: Any = lagged_f0, name: str = "") -> Pipeline:
    config = dataclasses.replace(pipeline.config, columns=FEATURES + ("f0_lag",))
    return dataclasses.replace(
        pipeline, features=features, config=config, name=name or pipeline.name + "_lag"
    )


FLAKY_CALLS: list[int] = []
SEEN_ROWS: list[int] = []


def peeking_fit(dataset: Dataset, train: Segments, config: RidgeConfig) -> RidgeModel:
    """Records how many rows it could see; the guard test reads it."""
    SEEN_ROWS.append(dataset.rows)
    return ridge_fit(dataset, train, config)


def flaky_fit(dataset: Dataset, train: Segments, config: RidgeConfig) -> RidgeModel:
    """Raises on its third call ever; the test clears ``FLAKY_CALLS`` to arm it."""
    FLAKY_CALLS.append(train[-1][1])
    if len(FLAKY_CALLS) == 3:
        raise RuntimeError("boom at the third fit")
    return ridge_fit(dataset, train, config)


def ridge_save(model: RidgeModel) -> bytes:
    return formats.arrays_save(
        {"weights": model.weights, "bias": np.array([model.bias])}
    )


def ridge_load(payload: bytes) -> RidgeModel:
    arrays = formats.arrays_load(payload)
    return RidgeModel(arrays["weights"], float(arrays["bias"][0]))


def scale(pred: np.ndarray, dataset: Dataset, rng: Range, factor: float) -> np.ndarray:
    return pred * factor


def pickle_save(model: RidgeModel) -> bytes:
    return pickle.dumps(model)


def pickle_load(payload: bytes) -> RidgeModel:
    return pickle.loads(payload)


@dataclasses.dataclass(frozen=True)
class BlendConfig:
    shrink: float = 0.0


@dataclasses.dataclass(frozen=True)
class BlendModel:
    weights: np.ndarray


def blend_fit(
    dataset: Dataset, train: Segments, config: BlendConfig, members: list[np.ndarray]
) -> BlendModel:
    """Least squares on the members' in-sample predictions, shrunk toward equal."""
    x = np.column_stack(members)
    y = dataset.column(TARGET, train)
    fitted = np.linalg.lstsq(x, y, rcond=None)[0]
    equal = np.full(len(members), 1.0 / len(members))
    return BlendModel((1 - config.shrink) * fitted + config.shrink * equal)


def blend_predict(
    model: BlendModel, dataset: Dataset, rng: Range, members: list[np.ndarray]
) -> np.ndarray:
    return np.column_stack(members) @ model.weights


def blend_save(model: BlendModel) -> bytes:
    return formats.arrays_save({"weights": model.weights})


def blend_load(payload: bytes) -> BlendModel:
    return BlendModel(formats.arrays_load(payload)["weights"])


def equal_fit(dataset: Dataset, train: Segments, config: BlendConfig) -> BlendModel:
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
        format=formats.Format.ARROW_ARRAYS,
        members=members,
        in_sample=True,
    )


def cheating_predict(model: RidgeModel, dataset: Dataset, rng: Range) -> np.ndarray:
    """Returns the target itself; the run hides it inside the window."""
    return dataset.column(TARGET, rng)


def peeking_predict(model: RidgeModel, dataset: Dataset, rng: Range) -> np.ndarray:
    """Adds the window's mean feature to every row: reads the future inside it."""
    return ridge_predict(model, dataset, rng) + dataset.column("f0", rng).mean()


def self_editing_fit(dataset: Dataset, train: Segments, config: RidgeConfig):
    """Appends a comment to its own module while fitting."""
    path = pathlib.Path(__file__)
    path.write_text(path.read_text() + "\n# edited during the run\n")
    return ridge_fit(dataset, train, config)


def zip_save(model: RidgeModel) -> bytes:
    """A portable zip: the arrow-arrays model plus a text note."""
    return formats.zip_save(
        {
            "model.arrow": (formats.Format.ARROW_ARRAYS, ridge_save(model)),
            "note.txt": (formats.Format.TEXT, b"ridge"),
        }
    )


def zip_pickle_save(model: RidgeModel) -> bytes:
    return formats.zip_save({"model.pkl": (formats.Format.PICKLE, pickle_save(model))})


def zip_load(payload: bytes) -> RidgeModel:
    return ridge_load(formats.zip_load(payload)["model.arrow"])


def zip_pickle_load(payload: bytes) -> RidgeModel:
    return pickle_load(formats.zip_load(payload)["model.pkl"])


@dataclasses.dataclass(frozen=True)
class SimConfig:
    cost: float


def sim_metrics(series: np.ndarray) -> dict[str, float]:
    return {"pnl": float(series[:, 0].sum()), "turnover": float(series[:, 1].sum())}


def sign_sim(
    pred: np.ndarray, dataset: Dataset, rng: Range, config: SimConfig
) -> np.ndarray:
    truth = dataset.column(TARGET, rng)
    position = np.sign(pred)
    flips = np.abs(np.diff(position, prepend=0.0))
    return np.column_stack([position * truth - config.cost * flips, flips])


sign_scorer = Scorer(
    sign_sim,
    metrics=sim_metrics,
    directions={"pnl": "max", "turnover": "min"},
    columns=("pnl", "flips"),
)


def ridge(window_months: int, alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        name=f"ridge_{window_months}m",
        fit=ridge_fit,
        predict=ridge_predict,
        save=ridge_save,
        load=ridge_load,
        config=RidgeConfig(window_months, alpha),
        format=formats.Format.ARROW_ARRAYS,
    )


def evaluation(dataset: str, cost: float = 0.001, split: Any = None) -> Evaluation:
    return Evaluation(
        dataset=dataset,
        split=split
        or CalendarWalkForward(
            first_cutoff="2025-05-01",
            embargo_timestamps=1,
            min_folds=3,
        ),
        scorer=sign_scorer,
        config=SimConfig(cost=cost),
    )


def flaky(window_months: int = 3) -> Pipeline:
    return dataclasses.replace(ridge(window_months), fit=flaky_fit, name="flaky")


pipelines = [ridge(1), ridge(3), ridge(6)]


def evaluations(dataset: str) -> list[Evaluation]:
    return [evaluation(dataset)]


def centre_window(pred: np.ndarray, dataset: Dataset, rng: Range, config) -> np.ndarray:
    """Subtracts the window's mean prediction: reads later predictions."""
    return pred - pred.mean()


def biasless_save(model: RidgeModel) -> bytes:
    """Drops the bias, as a writer that loses state would."""
    return formats.arrays_save({"weights": model.weights, "bias": np.zeros(1)})


def demean_at_timestamp(model: Any, dataset: Dataset, rng: Range) -> np.ndarray:
    """Each row's f0 minus its timestamp's mean: a same-time cross-section."""
    _, t = np.unique(dataset.ts[rng[0] : rng[1]], return_inverse=True)
    x = dataset.column("f0", rng)
    return x - (np.bincount(t, x) / np.bincount(t))[t]


def day_mean(dataset: Dataset) -> dict[str, np.ndarray]:
    """Each row's f0 minus its day's mean: reads later rows of the same day."""
    days, d = np.unique(dataset.ts.astype("datetime64[D]"), return_inverse=True)
    x = dataset.columns["f0"]
    return {"f0_lag": x - (np.bincount(d, x) / np.bincount(d))[d]}


def panel(ledger: Ledger, reveal: Any = None) -> str:
    """Two rows a minute; the target is f0 two minutes on."""
    f0, f1, f2 = np.random.default_rng(3).standard_normal((3, 400))
    ts = np.datetime64("2025-01-01", "ns") + (np.arange(400) // 2).astype(
        "timedelta64[m]"
    )
    columns = {"f0": f0, "f1": f1, "f2": f2, TARGET: np.r_[f0[4:], np.zeros(4)]}
    return record(
        ledger,
        Dataset(columns, ts),
        source="panel",
        params={},
        filters=(),
        targets=(TARGET,),
        reveal=reveal,
    )


def label_two_minutes_back(dataset: Dataset) -> dict[str, np.ndarray]:
    return {"f0_lag": np.r_[np.zeros(4), dataset.columns[TARGET][:-4]]}


def label_one_minute_back(dataset: Dataset) -> dict[str, np.ndarray]:
    return {"f0_lag": np.r_[np.full(2, np.nan), dataset.columns[TARGET][:-2]]}


def lagged_f1(dataset: Dataset) -> dict[str, np.ndarray]:
    return {"f1_lag": np.concatenate([[np.nan], dataset.columns["f1"][:-1]])}


def jittery_predict(model: RidgeModel, dataset: Dataset, rng: Range) -> np.ndarray:
    """Ridge plus a rounding-sized wobble that depends on how many rows it sees."""
    p = ridge_predict(model, dataset, rng)
    return p + 1e-12 * np.abs(p).max() * (dataset.rows % 3)


def add_f0_lag(pred: np.ndarray, dataset: Dataset, rng: Range, config) -> np.ndarray:
    """Reads a feature column from the postprocess: inherited from the members."""
    return pred + np.nan_to_num(dataset.column("f0_lag", rng))


def label_one_date_back(dataset: Dataset) -> dict[str, np.ndarray]:
    """The previous trading date's label at the same slot: known from the next date."""
    return {"f0_lag": np.r_[np.full(20, np.nan), dataset.columns[TARGET][:-20]]}


def label_one_row_back(dataset: Dataset) -> dict[str, np.ndarray]:
    """The previous row's label: same date, so not yet known under a one-date lag."""
    return {"f0_lag": np.r_[np.nan, dataset.columns[TARGET][:-1]]}


def label_memory_fit(
    dataset: Dataset, train: Segments, config: RidgeConfig
) -> RidgeModel:
    """Memorizes every label it can see, inside its train segments or not."""
    return RidgeModel(dataset.column(TARGET, (0, dataset.rows)), 0.0)


def label_replay_predict(model: RidgeModel, dataset: Dataset, rng: Range) -> np.ndarray:
    """Replays the memorized labels over ``rng``, zero past them or where blank."""
    seen = np.zeros(rng[1] - rng[0])
    known = model.weights[rng[0] : rng[1]]
    seen[: len(known)] = np.nan_to_num(known)
    return seen
