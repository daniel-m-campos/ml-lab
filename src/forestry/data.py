"""Freezing a dataset and loading it back.

A dataset is the sampled rows of one (process, instrument, window) passed through filter steps,
with named target columns. Its id covers the whole recipe; the row covers the bytes.

Examples
--------
>>> dataset = freeze(ledger, session, process="toy", params={}, instrument="X",
...                  filters=(), targets=("ret_1",))  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from forestry import hashing
from forestry.ledger import Kinds, Ledger
from forestry.session import Session


def freeze(
    ledger: Ledger,
    session: Session,
    *,
    process: str,
    params: Mapping[str, Any],
    instrument: str,
    filters: Sequence[Callable],
    targets: Sequence[str],
) -> str:
    """Apply the filters, check the targets, store the rows; returns the dataset id."""
    window = [str(session.start), str(session.end_exclusive)]
    recipe = {
        "process": process,
        "params": params,
        "instrument": instrument,
        "window": window,
        "filters": list(filters),
        "targets": list(targets),
    }
    dataset_id = hashing.content_hash(recipe)
    if ledger.get(Kinds.DATASET, dataset_id) is not None:
        return dataset_id
    for filt in filters:
        session = filt(session)
    missing = [t for t in targets if t not in session.columns]
    if missing:
        raise KeyError(f"targets not in session: {missing}")
    ledger.put(
        Kinds.DATASET,
        dataset_id,
        {**recipe, "rows": session.rows, "blob": ledger.put_blob(session.to_bytes())},
    )
    return dataset_id


def session(ledger: Ledger, dataset_id: str) -> Session:
    """Load a dataset's rows into a resident session."""
    row = ledger.get(Kinds.DATASET, dataset_id)
    if row is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return Session.from_bytes(ledger.get_blob(row["blob"]))
