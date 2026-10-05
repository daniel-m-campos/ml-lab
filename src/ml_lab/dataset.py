"""Recording a dataset and loading it back.

A dataset is the rows of one (source, params, window) passed through filter steps,
with named target columns. Its id is the data: column names, dtypes and bytes, so a
loader fix that changes rows is a new dataset and an edit that changes nothing is not.
The recipe rides on the event as provenance; the bytes are Parquet.

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
    reveal: Mapping[str, datetime.timedelta | int] | None = None,
) -> str:
    """Apply the filters, check the targets, store the rows; returns the dataset id.

    ``reveal`` maps a target to when its label is known: a ``timedelta`` after its
    row's timestamp, or an ``int`` of later dates with rows (``1`` is the next
    trading day's open, ``2`` two trading dates later whatever the calendar gap). A
    features step may read a revealed target, probed against that lag. Targets and
    their lags are part of the dataset id.
    """
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


def _lag(reveal: datetime.timedelta | int) -> float | dict[str, int]:
    """Seconds for a time lag, ``{"dates": n}`` for a count of later dates."""
    if isinstance(reveal, datetime.timedelta):
        return reveal.total_seconds()
    return {"dates": int(reveal)}


def data_id(session: Session, targets: Mapping[str, Any] | None = None) -> str:
    """The content id of a session: its column names, dtypes and bytes, and which
    columns are targets with their reveal lags in seconds.
    """
    ts = None if session.ts is None else identity.bytes_hash(session.ts.tobytes())
    columns = {
        name: [str(a.dtype), identity.bytes_hash(np.ascontiguousarray(a).tobytes())]
        for name, a in session.columns.items()
    }
    return identity.content_hash({"ts": ts, "columns": columns, "targets": targets})


def load(ledger: Ledger, dataset_id: str) -> Session:
    """Load a dataset's rows into a resident session."""
    event = ledger.latest(Event.DATASET, dataset_id)
    if event is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return formats.session_load(ledger.blobs / event["payload"]["blob"]["sha"])
