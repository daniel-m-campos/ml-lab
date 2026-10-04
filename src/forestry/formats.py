"""Every byte conversion the ledger writes, in formats that open without this
environment.

Datasets and predictions are Parquet. A dict of arrays is one Arrow IPC file. Libraries
with a path-only API round-trip through a temp file.

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
import pyarrow as pa
import pyarrow.parquet as pq

from forestry.session import Session

TS = "ts"
PREDICTION = "prediction"


class Format:
    PARQUET = "parquet"
    ARROW_ARRAYS = "arrow-arrays"
    DIFF = "text/x-diff"
    TEXT = "text/plain"


# Sessions =============================================================================


def session_save(session: Session) -> bytes:
    """A session as one Parquet file: ``ts`` timestamp[s] plus one column per array."""
    stamps = {} if session.ts is None else {TS: pa.array(session.ts, pa.timestamp("s"))}
    return _parquet_bytes(pa.table({**stamps, **session.columns}))


def session_load(payload: bytes) -> Session:
    return session_from_table(pq.read_table(io.BytesIO(payload)))


def session_from_table(table: pa.Table, ts: str = TS) -> Session:
    """A session from an Arrow table; a ``ts`` timestamp column is optional."""
    has_ts = ts in table.column_names
    stamps = table[ts].to_numpy().astype("datetime64[s]") if has_ts else None
    columns = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name != ts
    }
    return Session(columns, stamps)


# Series ===============================================================================


def series_save(values: np.ndarray) -> bytes:
    """A 1-D float array as a one-column Parquet file."""
    return _parquet_bytes(pa.table({PREDICTION: np.asarray(values, dtype=np.float64)}))


def series_load(payload: bytes) -> np.ndarray:
    return pq.read_table(io.BytesIO(payload))[PREDICTION].to_numpy()


# Arrays ===============================================================================


def arrays_save(arrays: dict[str, np.ndarray]) -> bytes:
    """A dict of float arrays as an Arrow IPC file: one row per array with its name and
    shape.
    """
    names = list(arrays)
    table = pa.table(
        {
            "name": names,
            "shape": [list(np.asarray(arrays[n]).shape) for n in names],
            "data": [np.asarray(arrays[n], dtype=np.float64).ravel() for n in names],
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def arrays_load(payload: bytes) -> dict[str, np.ndarray]:
    table = pa.ipc.open_file(pa.BufferReader(payload)).read_all()
    rows = zip(
        table["name"].to_pylist(),
        table["shape"].to_pylist(),
        table["data"].to_pylist(),
        strict=True,
    )
    return {
        name: np.asarray(data, dtype=np.float64).reshape(shape)
        for name, shape, data in rows
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


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()
