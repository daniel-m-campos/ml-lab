"""Reading the Optiver close auction CSV and freezing it as a dataset."""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import polars as pl

from ml_lab.dataset import record
from ml_lab.experiment import step
from ml_lab.ledger import Ledger
from ml_lab.session import Session

TRAIN_CSV = pathlib.Path("data/optiver/optiver-trading-at-the-close/train.csv")
BASE_DATE = datetime.date(2021, 1, 4)
CLOSE_AUCTION_START_S = 15 * 3600 + 50 * 60
SOURCE = "optiver-close-2023"
INSTRUMENT = "nasdaq-close-auction"
TARGET = "target"
DROPPED = ("row_id", "time_id")


def available() -> bool:
    return TRAIN_CSV.is_file()


def load(stocks: tuple[int, ...] | None, path: pathlib.Path = TRAIN_CSV) -> Session:
    """Read the competition CSV onto a synthetic calendar axis."""
    frame = pl.read_csv(path)
    if stocks is not None:
        frame = frame.filter(pl.col("stock_id").is_in(list(stocks)))
    seconds = (
        frame["date_id"].to_numpy() * 86_400
        + CLOSE_AUCTION_START_S
        + frame["seconds_in_bucket"].to_numpy()
    )
    order = np.lexsort((frame["stock_id"].to_numpy(), seconds))
    ts = np.datetime64(BASE_DATE, "s") + seconds.astype("timedelta64[s]")
    floats = frame.drop(DROPPED).cast(pl.Float64)
    return Session({c: floats[c].to_numpy()[order] for c in floats.columns}, ts[order])


@step
def drop_null_target(session: Session) -> Session:
    keep = ~np.isnan(session.columns[TARGET])
    return Session({k: v[keep] for k, v in session.columns.items()}, session.ts[keep])


def dataset(ledger: Ledger, stocks: str = "all") -> str:
    """Ingest a stock subset with null targets dropped; ``stocks`` is "all", "20" or
    "0,1,2".
    """
    selected = parse_stocks(stocks)
    return record(
        ledger,
        load(selected),
        source=SOURCE,
        params={
            "stocks": list(selected) if selected else "all",
            "base_date": str(BASE_DATE),
            "instrument": INSTRUMENT,
        },
        filters=(drop_null_target,),
        targets=(TARGET,),
    )


def parse_stocks(spec: str) -> tuple[int, ...] | None:
    if spec == "all":
        return None
    if "," in spec:
        return tuple(int(s) for s in spec.split(","))
    return tuple(range(int(spec)))
