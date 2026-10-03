"""Capture and dataset for the Optiver close auction data."""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv

from forestry import data
from forestry.declare import step
from forestry.ledger import Ledger
from forestry.session import Frame

TRAIN_CSV = pathlib.Path("data/optiver/optiver-trading-at-the-close/train.csv")
BASE_DATE = datetime.date(2021, 1, 4)
CLOSE_AUCTION_START_S = 15 * 3600 + 50 * 60
PROCESS = "optiver-close-2023"
INSTRUMENT = "nasdaq-close-auction"
TARGET = "target"
DROPPED = ("row_id", "time_id")


def available() -> bool:
    return TRAIN_CSV.is_file()


def load_frame(stocks: tuple[int, ...] | None, path: pathlib.Path = TRAIN_CSV) -> Frame:
    """Read the competition CSV into a frame on a synthetic calendar axis."""
    table = pyarrow.csv.read_csv(path)
    if stocks is not None:
        table = table.filter(pc.is_in(table["stock_id"], pa.array(stocks)))
    seconds = (
        table["date_id"].to_numpy() * 86_400
        + CLOSE_AUCTION_START_S
        + table["seconds_in_bucket"].to_numpy()
    )
    order = np.lexsort((table["stock_id"].to_numpy(), seconds))
    ts = (np.datetime64(BASE_DATE, "s") + seconds[order].astype("timedelta64[s]")).astype(
        "datetime64[s]"
    )
    columns = {
        name: table[name].to_numpy(zero_copy_only=False).astype(np.float64)[order]
        for name in table.column_names
        if name not in DROPPED
    }
    return Frame(ts, columns)


@step
def drop_null_target(frame: Frame) -> Frame:
    keep = ~np.isnan(frame.columns[TARGET])
    return Frame(frame.ts[keep], {k: v[keep] for k, v in frame.columns.items()})


def freeze(ledger: Ledger, stocks: tuple[int, ...] | None) -> str:
    """Freeze the capture for a stock subset and the dataset with null targets dropped."""
    frame = load_frame(stocks)
    capture = data.freeze_capture(
        ledger,
        frame,
        process=PROCESS,
        params={"stocks": list(stocks) if stocks else "all", "base_date": str(BASE_DATE)},
        instrument=INSTRUMENT,
    )
    return data.freeze_dataset(ledger, capture, filters=(drop_null_target,), targets=(TARGET,))
