"""Reading the Optiver close auction CSV and freezing it as a dataset."""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv

from forestry import formats
from forestry.dataset import record
from forestry.experiment import step
from forestry.ledger import Ledger
from forestry.session import Session

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
    table = pyarrow.csv.read_csv(path)
    if stocks is not None:
        table = table.filter(pc.is_in(table["stock_id"], pa.array(stocks)))
    seconds = (
        table["date_id"].to_numpy() * 86_400
        + CLOSE_AUCTION_START_S
        + table["seconds_in_bucket"].to_numpy()
    )
    order = pa.array(np.lexsort((table["stock_id"].to_numpy(), seconds)))
    ts = np.datetime64(BASE_DATE, "s") + seconds.astype("timedelta64[s]")
    kept = table.drop_columns(list(DROPPED)).append_column(
        "ts", pa.array(ts, pa.timestamp("s"))
    )
    floats = pa.table(
        {
            name: pc.cast(kept[name], pa.float64()) if name != "ts" else kept[name]
            for name in kept.column_names
        }
    )
    return formats.session_from_table(floats.take(order))


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
