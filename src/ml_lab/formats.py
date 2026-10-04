"""Every byte conversion the ledger writes, in formats that open without this
environment.

Datasets and predictions are Parquet. A dict of arrays is one Arrow IPC file. Both are
written and read by polars, the library a researcher already has open for the data
itself. Libraries with a path-only API round-trip through a temp file.

Examples
--------
>>> import numpy as np
>>> arrays_load(arrays_save({"w": np.arange(3.0)}))["w"]
array([0., 1., 2.])
"""

from __future__ import annotations

import io
import pathlib
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from ml_lab.session import Session

TS = "ts"
PREDICTION = "prediction"


class Format:
    PARQUET = "parquet"
    ARROW_ARRAYS = "arrow-arrays"
    BONSAI_MSGPACK = "bonsai-msgpack"
    DIFF = "text/x-diff"
    TEXT = "text/plain"
    PICKLE = "pickle"


KNOWN: dict[str, bool] = {
    Format.PARQUET: True,
    Format.ARROW_ARRAYS: True,
    Format.BONSAI_MSGPACK: True,
    Format.DIFF: True,
    Format.TEXT: True,
    Format.PICKLE: False,
}
"""Format name to whether it opens without this Python environment. A save step must
declare one of these; add an entry to admit a new format."""


# Sessions =============================================================================


def session_save(session: Session) -> bytes:
    """A session as one Parquet file: ``ts`` as a nanosecond timestamp plus one column
    per array.
    """
    stamps = {} if session.ts is None else {TS: session.ts}
    return _parquet_bytes(pl.DataFrame({**stamps, **session.columns}))


def session_load(payload: bytes) -> Session:
    return session_from_frame(pl.read_parquet(io.BytesIO(payload)))


def session_from_frame(frame: pl.DataFrame, ts: str = TS) -> Session:
    """A session from a frame; a ``ts`` timestamp column is optional."""
    stamps = frame[ts].to_numpy() if ts in frame.columns else None
    columns = {name: frame[name].to_numpy() for name in frame.columns if name != ts}
    return Session(columns, stamps)


# Series ===============================================================================


def series_save(values: np.ndarray) -> bytes:
    """A 1-D float array as a one-column Parquet file."""
    return _parquet_bytes(pl.DataFrame({PREDICTION: np.asarray(values, np.float64)}))


def series_load(payload: bytes) -> np.ndarray:
    return pl.read_parquet(io.BytesIO(payload))[PREDICTION].to_numpy()


# Arrays ===============================================================================


def arrays_save(arrays: dict[str, np.ndarray]) -> bytes:
    """A dict of float arrays as an Arrow IPC file: one row per array with its name and
    shape.
    """
    names = list(arrays)
    frame = pl.DataFrame(
        [
            pl.Series("name", names),
            pl.Series(
                "shape", [list(np.shape(arrays[n])) for n in names], pl.List(pl.Int64)
            ),
            pl.Series(
                "data",
                [np.asarray(arrays[n], np.float64).ravel() for n in names],
                pl.List(pl.Float64),
            ),
        ]
    )
    sink = io.BytesIO()
    frame.write_ipc(sink)
    return sink.getvalue()


def arrays_load(payload: bytes) -> dict[str, np.ndarray]:
    frame = pl.read_ipc(io.BytesIO(payload))
    return {
        name: np.asarray(data, np.float64).reshape(shape)
        for name, shape, data in frame.iter_rows()
    }


# Path-only libraries ==================================================================


def bytes_via_file(write: Callable[[str], Any], suffix: str) -> bytes:
    """Bytes from a library that can only write to a path."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        write(handle.name)
        return pathlib.Path(handle.name).read_bytes()


def load_via_file[T](payload: bytes, read: Callable[[str], T], suffix: str) -> T:
    """An object from a library that can only read from a path."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(payload)
        handle.flush()
        return read(handle.name)


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    sink = io.BytesIO()
    frame.write_parquet(sink)
    return sink.getvalue()
