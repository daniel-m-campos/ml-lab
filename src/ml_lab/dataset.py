"""Recording a dataset and loading it back.

A dataset is the rows of one (source, params, window) passed through filter steps,
with named target columns. Its id is the data: the ``ts`` bytes, column names, dtypes
and values, so a loader fix that changes rows is a new dataset and an edit that
changes nothing is not. The recipe rides on the event as provenance; the bytes are
Parquet.

Examples
--------
>>> dataset = record(ledger, session, source="toy", params={},
...                  filters=(), targets=("ret_1",))  # doctest: +SKIP
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ml_lab import dates, formats, identity
from ml_lab.ledger import Event, Ledger
from ml_lab.session import Session


def record(
    ledger: Ledger,
    session: Session,
    *,
    source: str,
    params: Mapping[str, Any],
    filters: Sequence[Callable],
    targets: Sequence[str],
    reveal: Mapping[str, datetime.timedelta | np.timedelta64 | int] | None = None,
) -> str:
    """Apply the filters, check the targets, store the rows; returns the dataset id.

    ``reveal`` maps a target to when its label is known: a ``timedelta`` or
    ``numpy.timedelta64`` after its row's timestamp, or an ``int`` of later dates with
    rows (``1`` is the next trading day's open, ``2`` two trading dates later whatever
    the calendar gap). A features step may read a revealed target, probed against
    that lag. Targets and their lags are part of the dataset id.
    """
    if session.rows == 0:
        raise ValueError("a dataset needs at least one row")
    has_clock = session.ts is not None
    reveal = dict(reveal or {})
    if reveal and not has_clock:
        raise ValueError("a reveal lag needs a session with timestamps")
    if set(reveal) - set(targets):
        raise KeyError(
            f"reveal names non-targets: {sorted(set(reveal) - set(targets))}"
        )
    labels = {t: _lag(reveal[t]) if t in reveal else None for t in targets}
    window = [str(d) for d in dates.span(session)] if has_clock else None
    recipe = {
        "source": source,
        "params": params,
        "window": window,
        "filters": list(filters),
        "targets": labels,
    }
    for filt in filters:
        session = filt(session)
    missing = [t for t in targets if t not in session.columns]
    if missing:
        raise KeyError(f"targets not in session: {missing}")
    text = [t for t in targets if not _numeric(session.columns[t].dtype)]
    if text:
        raise TypeError(
            f"targets {text} are not numeric; targets are masked with NaN, so encode "
            "labels as numbers"
        )
    dataset_id = data_id(session, labels)
    if ledger.latest(Event.DATASET, dataset_id) is not None:
        return dataset_id
    sha = ledger.put_blob(formats.session_save(session))
    payload = {
        "source": source,
        "window": window,
        "recipe": identity.canonical(recipe),
        "rows": session.rows,
        "blob": {"sha": sha, "format": formats.Format.PARQUET},
    }
    ledger.append(Event.DATASET, dataset_id, dataset_id, payload, id=dataset_id)
    return dataset_id


def _numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _lag(reveal: datetime.timedelta | np.timedelta64 | int) -> float | dict[str, int]:
    """Seconds for a time lag, ``{"dates": n}`` for a count of later dates."""
    time = isinstance(reveal, (datetime.timedelta, np.timedelta64))
    if not time and not isinstance(reveal, (int, np.integer)):
        raise TypeError(
            f"a reveal lag is a timedelta or an int, not {type(reveal).__name__}"
        )
    lag = reveal / np.timedelta64(1, "s") if time else reveal
    if lag < 0:
        raise ValueError(f"a reveal lag cannot be negative: {reveal!r}")
    return lag if time else {"dates": lag}


def data_id(session: Session, targets: Mapping[str, Any] | None = None) -> str:
    """The content id of a session: its ``ts`` bytes, column names, dtypes and values
    (a string column by its strings), and which columns are targets with their reveal
    lags in seconds or a count of dates.
    """
    ts = None if session.ts is None else identity.bytes_hash(session.ts.tobytes())
    columns = {
        name: [str(a.dtype), identity.array_hash(a)]
        for name, a in session.columns.items()
    }
    return identity.content_hash({"ts": ts, "columns": columns, "targets": targets})


def load(ledger: Ledger, dataset_id: str) -> Session:
    """Load a dataset's rows into a resident session."""
    event = ledger.latest(Event.DATASET, dataset_id)
    if event is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return formats.session_load(ledger.blobs / event["payload"]["blob"]["sha"])
