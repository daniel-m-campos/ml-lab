"""Every byte conversion the ledger writes, in formats that open without this
environment.

Datasets (through ``ml_lab.dataset``) and predictions are Parquet. A dict of arrays is
one Arrow IPC file. Both are written and read by polars, the library a researcher
already has open for the data itself. Libraries with a path-only API round-trip through
a temp file; ``pickle_save`` and ``pickle_load`` serve an sklearn model, marked not
portable.

Examples
--------
>>> import numpy as np
>>> arrays_load(arrays_save({"w": np.arange(3.0)}))["w"]
array([0., 1., 2.])
"""

from __future__ import annotations

import io
import json
import pathlib
import pickle
import tempfile
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

TS = "ts"
PREDICTION = "prediction"


class Format:
    PARQUET = "parquet"
    ARROW_ARRAYS = "arrow-arrays"
    BONSAI_MSGPACK = "bonsai-msgpack"
    DIFF = "text/x-diff"
    TEXT = "text/plain"
    PICKLE = "pickle"
    ZIP = "zip"


KNOWN: dict[str, bool] = {
    Format.PARQUET: True,
    Format.ARROW_ARRAYS: True,
    Format.BONSAI_MSGPACK: True,
    Format.DIFF: True,
    Format.TEXT: True,
    Format.PICKLE: False,
    Format.ZIP: True,
}
"""Format name to whether it opens without this Python environment. A pipeline's
``format`` must be one of these; add an entry to admit a new format. A ``zip`` model is
portable iff every member is; see ``zip_save``."""

MANIFEST = "formats.json"


# Zip containers =======================================================================


def zip_save(parts: dict[str, tuple[str, bytes]]) -> bytes:
    """Named byte members, each with its format from ``KNOWN``, as one deterministic
    zip with a ``formats.json`` manifest: a model that is a foreign file plus a few
    numbers.
    """
    if MANIFEST in parts:
        raise ValueError(
            f"{MANIFEST} is the manifest's name; name the member otherwise"
        )
    manifest = json.dumps({n: f for n, (f, _) in parts.items()}, sort_keys=True)
    members = [
        (MANIFEST, manifest.encode()),
        *sorted((n, d) for n, (_, d) in parts.items()),
    ]
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w") as archive:
        for name, data in members:
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)
    return sink.getvalue()


def zip_load(payload: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {n: archive.read(n) for n in archive.namelist() if n != MANIFEST}


def zip_formats(payload: bytes) -> dict[str, str]:
    """Member name to declared format, from the manifest."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return json.loads(archive.read(MANIFEST))


# Series ===============================================================================


def series_save(values: np.ndarray) -> bytes:
    """Predictions, ``(rows,)`` or ``(rows, k)``, as Parquet: one column
    ``prediction``, or ``prediction_0`` to ``prediction_{k-1}``.
    """
    values = np.asarray(values, np.float64)
    columns = (
        {PREDICTION: values}
        if values.ndim == 1
        else {f"{PREDICTION}_{i}": c for i, c in enumerate(values.T)}
    )
    return parquet_bytes(pl.DataFrame(columns))


def series_load(payload: bytes) -> np.ndarray:
    frame = pl.read_parquet(io.BytesIO(payload))
    if PREDICTION in frame.columns:
        return frame[PREDICTION].to_numpy()
    return frame.to_numpy()


# Arrays ===============================================================================


def arrays_save(arrays: dict[str, np.ndarray]) -> bytes:
    """A dict of float arrays as an Arrow IPC file: one row per array with its name and
    shape.
    """
    names = list(arrays)
    data = [np.asarray(arrays[n], np.float64).ravel() for n in names]
    if not any(d.size for d in data):
        data = [d.tolist() for d in data]
    frame = pl.DataFrame(
        [
            pl.Series("name", names),
            pl.Series(
                "shape", [list(np.shape(arrays[n])) for n in names], pl.List(pl.Int64)
            ),
            pl.Series("data", data, pl.List(pl.Float64)),
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


# Pickle ===============================================================================


def pickle_save(obj: Any) -> bytes:
    """Any object as pickle bytes, for a ``pickle`` model; not portable."""
    return pickle.dumps(obj)


def pickle_load(payload: bytes) -> Any:
    return pickle.loads(payload)


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


def parquet_bytes(frame: pl.DataFrame) -> bytes:
    sink = io.BytesIO()
    frame.write_parquet(sink)
    return sink.getvalue()
