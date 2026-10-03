"""Freezing captures and datasets, and loading a dataset into a session.

A capture is the sampled bytes of one (process, instrument, window). A dataset is a capture
passed through filter steps with named target columns; its family is (process, instrument).

Examples
--------
>>> cap = freeze_capture(ledger, frame, process="toy", params={}, instrument="X")  # doctest: +SKIP
>>> dataset = freeze_dataset(ledger, capture, filters=(), targets=("ret_1",))  # doctest: +SKIP
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from forestry import hashing
from forestry.ledger import Kinds, Ledger
from forestry.session import Frame, Session

# Public Functions =================================================================================


def freeze_capture(
    ledger: Ledger, frame: Frame, *, process: str, params: Mapping[str, Any], instrument: str
) -> str:
    """Store a frame as a capture; the id covers the recipe, the row covers the bytes."""
    window = [str(frame.start), str(frame.end_exclusive)]
    capture_id = hashing.content_hash(
        {"process": process, "params": params, "instrument": instrument, "window": window}
    )
    blob = ledger.put_blob(_frame_bytes(frame))
    ledger.put(
        Kinds.CAPTURE,
        capture_id,
        {
            "process": process,
            "params": params,
            "instrument": instrument,
            "window": window,
            "blob": blob,
            "rows": frame.rows,
        },
    )
    return capture_id


def freeze_dataset(
    ledger: Ledger, capture_id: str, *, filters: Sequence[Callable], targets: Sequence[str]
) -> str:
    """Apply filter steps to a capture and name its targets; returns the dataset id."""
    capture = ledger.get(Kinds.CAPTURE, capture_id)
    if capture is None:
        raise KeyError(f"capture {capture_id} not found")
    dataset_id = hashing.content_hash(
        {"capture": capture_id, "filters": list(filters), "targets": list(targets)}
    )
    if ledger.get(Kinds.DATASET, dataset_id) is not None:
        return dataset_id
    frame = _frame_from_bytes(ledger.get_blob(capture["blob"]))
    for filt in filters:
        frame = filt(frame)
    missing = [t for t in targets if t not in frame.columns]
    if missing:
        raise KeyError(f"targets not in frame: {missing}")
    ledger.put(
        Kinds.DATASET,
        dataset_id,
        {
            "capture": capture_id,
            "filters": list(filters),
            "targets": list(targets),
            "family": [capture["process"], capture["instrument"]],
            "window": capture["window"],
            "blob": ledger.put_blob(_frame_bytes(frame)),
            "rows": frame.rows,
        },
    )
    return dataset_id


def session(ledger: Ledger, dataset_id: str) -> Session:
    """Load a dataset's frame into a resident session."""
    row = ledger.get(Kinds.DATASET, dataset_id)
    if row is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return Session(_frame_from_bytes(ledger.get_blob(row["blob"])))


# Private Functions ================================================================================


def _frame_bytes(frame: Frame) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, __ts__=frame.ts.astype("datetime64[s]"), **frame.columns)
    return buf.getvalue()


def _frame_from_bytes(payload: bytes) -> Frame:
    with np.load(io.BytesIO(payload)) as npz:
        columns = {k: npz[k] for k in npz.files if k != "__ts__"}
        return Frame(npz["__ts__"], columns)
