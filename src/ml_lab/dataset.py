"""Datasets: columns over rows in a fixed order, recorded once and loaded back.

A ``Dataset`` is a table held resident, read by row ranges through ``column(name,
rows)`` and ``matrix(cols, rows)``, with a clock lookup when ``ts`` is given;
``from_frame`` builds one from a polars frame. A recorded dataset is the rows of one
(source, params), cleaned by the caller before ``record``, with named target columns.
Its id is the data: the ``ts`` bytes, column names, dtypes and values, so a loader fix
that changes rows is a new dataset and an edit that changes nothing is not. The recipe
rides on the event as provenance; the bytes are Parquet.

Examples
--------
>>> import numpy as np
>>> ts = np.array(["2025-01-01T00:00", "2025-01-02T00:00"], dtype="datetime64[s]")
>>> d = Dataset({"x": np.array([1.0, 2.0])}, ts)
>>> d.index_of("2025-01-02")
1
>>> dataset_id = record(ledger, d, source="toy", targets=("x",))  # doctest: +SKIP
"""

from __future__ import annotations

import datetime
import io
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from ml_lab import dates, formats, identity
from ml_lab.formats import TS
from ml_lab.identity import Refused
from ml_lab.ledger import Event, Ledger

Range = tuple[int, int]
Rows = Range | tuple[Range, ...]
Segments = tuple[Range, ...]
Lag = np.timedelta64 | int | None


class Dataset:
    """Columns over rows in a fixed order, with date lookups when ``ts`` is given.
    ``computed_columns`` names the columns a pipeline's features functions added, in
    order; they are not in ``columns`` but read on demand from the Parquet file in
    ``stores`` through ``column`` and ``matrix``, so a column no function names costs
    no memory.
    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        ts: np.ndarray | None = None,
        computed_columns: tuple[str, ...] = (),
        stores: dict[str, pathlib.Path] | None = None,
    ):
        self.columns = {
            n: a.astype("datetime64[ns]", copy=False) if a.dtype.kind == "M" else a
            for n, a in columns.items()
        }
        self.computed_columns = computed_columns
        self.stores = stores or {}
        self.cut: int | None = None
        self.keep: np.ndarray | None = None
        self.ts = None if ts is None else ts.astype("datetime64[ns]")
        if self.ts is not None and np.isnat(self.ts).any():
            raise ValueError("timestamps must not contain NaT")
        if self.ts is not None and np.any(self.ts[1:] < self.ts[:-1]):
            raise ValueError("timestamps must be sorted")

    def __repr__(self) -> str:
        return f"Dataset(rows={self.rows}, clock={self.ts is not None})"

    @property
    def rows(self) -> int:
        if self.cut is not None:
            return self.cut
        return int(next(iter(self.columns.values())).shape[0])

    def index_of(self, when: str | datetime.date | datetime.datetime) -> int:
        """First row at or after ``when``; a date means its midnight."""
        return int(np.searchsorted(self._clock(), np.datetime64(when, "ns"), "left"))

    def date_at(self, row: int) -> datetime.date:
        self._clock()
        return self.ts[row].astype("datetime64[D]").astype(datetime.date)

    def upto(self, row: int) -> Dataset:
        """The first ``row`` rows as a view: positions are unchanged, so ranges into
        the full dataset stay valid, and nothing after ``row`` can be read.
        """
        view = Dataset.__new__(Dataset)
        view.columns = {k: v[:row] for k, v in self.columns.items()}
        view.ts = None if self.ts is None else self.ts[:row]
        view.computed_columns = self.computed_columns
        view.stores = self.stores
        view.cut = min(row, self.rows)
        view.keep = None if self.keep is None else self.keep[:row]
        return view

    def masked(self, reveal: dict[str, Lag], at: int) -> Dataset:
        """A view whose target columns are NaN for every label not known at row
        ``at``: a label with a time lag ``L`` is known from ``ts + L``, one with an
        integer lag ``n`` from the first row of the n-th later date that has rows, one
        without a lag only after its own timestamp; without a clock, rows from ``at``
        on.
        """
        view = self.upto(self.rows)
        for name, lag in reveal.items():
            values = view.columns[name].astype(np.float64, copy=True)
            if self.ts is None:
                values[at:] = np.nan
            elif lag is None:
                values[self.ts >= self.ts[at]] = np.nan
            elif isinstance(lag, np.timedelta64):
                values[self.ts + lag > self.ts[at]] = np.nan
            else:
                day = np.unique(self.ts.astype("datetime64[D]"), return_inverse=True)[1]
                values[day + lag > day[at]] = np.nan
            view.columns[name] = values
        return view

    def train_view(self, train: Segments) -> Dataset:
        """The view a fit sees: every column, computed columns included, blank outside
        ``train``, so a fit reads no row of a range it is scored on. A blank is NaN, NaT
        or None by dtype; an integer or boolean column becomes float.
        """
        view = self.upto(self.rows)
        view.keep = row_mask(self.rows, train)
        view.columns = {n: _blank(a, view.keep) for n, a in view.columns.items()}
        return view

    def boundary(self, row: int, lo: int = 0) -> int | None:
        """The first row of ``row``'s timestamp, or of the next one when that is
        ``lo``; None when no timestamp starts inside ``(lo, rows)``. Without a clock
        every row is its own timestamp.
        """
        if self.ts is None:
            return row if lo < row < self.rows else None
        first = int(np.searchsorted(self.ts, self.ts[row], "left"))
        if first > lo:
            return first
        after = int(np.searchsorted(self.ts, self.ts[row], "right"))
        return after if after < self.rows else None

    def date_starts(self, lo: int, hi: int) -> np.ndarray:
        """The first row of each date among rows ``lo`` to ``hi``, then ``hi``; without
        a clock the rows are one date.
        """
        if self.ts is None:
            return np.array([lo, hi])
        day = self.ts[lo:hi].astype("datetime64[D]")
        return np.r_[lo + np.flatnonzero(np.r_[True, day[1:] != day[:-1]]), hi]

    def middle(self, lo: int, hi: int) -> int:
        """The first row of the middle timestamp among rows ``lo`` to ``hi``."""
        stamps = np.arange(lo, hi) if self.ts is None else self.ts[lo:hi]
        first = np.unique(stamps, return_index=True)[1]
        return lo + int(first[len(first) // 2])

    def matrix(self, cols: Sequence[str], rows: Rows | None = None) -> np.ndarray:
        """Column-stacked features over a range or segments, every visible row when
        ``rows`` is None, shape (rows, len(cols)), filled one column at a time so the
        matrix is the only copy.
        """
        parts = segments((0, self.rows) if rows is None else rows)
        first = self.column(cols[0], parts)
        dtype = np.result_type(first, *(self._dtype(c) for c in cols[1:]))
        out = np.empty((len(first), len(cols)), dtype)
        out[:, 0] = first
        for j, name in enumerate(cols[1:], 1):
            out[:, j] = self.column(name, parts)
        return out

    def column(self, name: str, rows: Rows | None = None) -> np.ndarray:
        """One column over a range or segments, every visible row when ``rows`` is
        None.
        """
        parts = segments((0, self.rows) if rows is None else rows)
        for lo, hi in parts:
            if lo < 0 or hi < lo:
                raise ValueError(f"segment ({lo}, {hi}) asked; one needs 0 <= lo <= hi")
        end = max(hi for _, hi in parts)
        if end > self.rows:
            raise ValueError(
                f"rows up to {end} asked, {self.rows} visible before the cutoff"
            )
        if name in self.columns:
            values = self.columns[name]
            return np.concatenate([values[lo:hi] for lo, hi in parts])
        if name not in self.stores:
            raise KeyError(
                f"{name!r} is not a column: dataset columns {sorted(self.columns)}, "
                f"computed columns {list(self.computed_columns)}"
            )
        scan = pl.scan_parquet(self.stores[name]).select(name)
        values = np.concatenate(
            [scan.slice(lo, hi - lo).collect()[name].to_numpy() for lo, hi in parts]
        )
        if self.keep is None:
            return values
        return _blank(values, np.concatenate([self.keep[lo:hi] for lo, hi in parts]))

    def _dtype(self, name: str) -> np.dtype:
        if name in self.columns:
            return self.columns[name].dtype
        schema = pl.read_parquet_schema(self.stores[name])
        dtype = pl.Series([], dtype=schema[name]).to_numpy().dtype
        if self.keep is None:
            return dtype
        return _blank(np.empty(0, dtype), np.empty(0, bool)).dtype

    def _clock(self) -> np.ndarray:
        if self.ts is None:
            raise Refused("dataset has no timestamps; use a row-based split")
        return self.ts


def segments(rows: Rows) -> tuple[Range, ...]:
    """A range or a tuple of ranges as a tuple of ranges."""
    return (rows,) if isinstance(rows[0], (int, np.integer)) else rows


def row_mask(size: int, rows: Rows) -> np.ndarray:
    """A bool mask of ``size`` rows, True inside the segments."""
    mask = np.zeros(size, bool)
    for lo, hi in segments(rows):
        mask[lo:hi] = True
    return mask


def stratified_order(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """A row permutation that shuffles each class and interleaves the classes in
    proportion, so every contiguous block of the reordered rows holds each class in
    its share: rows sort by rank within their class over the class size, ties by
    class. Reorder the rows before ``record`` and a ``BlockedKFold`` is stratified.
    """
    classes = np.unique(y, return_inverse=True)[1]
    shuffled = np.random.default_rng(seed).permutation(len(classes))
    owner = classes[shuffled]
    by_class = np.argsort(owner, kind="stable")
    grouped = owner[by_class]
    rank = np.empty(len(owner))
    rank[by_class] = np.arange(len(owner)) - np.searchsorted(grouped, grouped)
    return shuffled[np.lexsort((owner, rank / np.bincount(owner)[owner]))]


def _blank(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """A copy of ``values`` blank where ``keep`` is False."""
    kind = values.dtype.kind
    text, clock = kind in "OSU", kind in "mM"
    dtype = object if text else values.dtype if clock or kind in "fc" else np.float64
    out = values.astype(dtype, copy=True)
    out[~keep] = None if text else "NaT" if clock else np.nan
    return out


# Recording ============================================================================


def record(
    ledger: Ledger,
    dataset: Dataset,
    *,
    source: str,
    targets: Sequence[str],
    params: Mapping[str, Any] | None = None,
    reveal: Mapping[str, datetime.timedelta | np.timedelta64 | int] | None = None,
    sealed_from: int | str | None = None,
) -> str:
    """Check the targets and store the rows; returns the dataset id.

    ``source`` and ``params`` name how the rows were made, cleaning included, so clean
    before ``record`` and name the cleaning in ``params``. ``reveal`` maps a target to
    when its label is known: a ``timedelta`` or ``numpy.timedelta64`` after its row's
    timestamp, or an ``int`` of later dates with rows (``1`` is the next trading day's
    open, ``2`` two trading dates later whatever the calendar gap). A features function
    may read a revealed target, probed against that lag. ``sealed_from``, a row (the
    first row of its timestamp) or a date (its first row at or after it), starts the
    sealed tail that only an ``Evaluation(sealed=True)`` validates on, each pipeline
    once, and that no fold trains on before such an evaluation tests it. Targets,
    their lags and the sealed tail's first row are part of the dataset id. An empty
    ``targets`` is refused.
    """
    if not targets:
        raise Refused(
            "a dataset declares at least one target; without one nothing is masked"
        )
    if dataset.rows == 0:
        raise ValueError("a dataset needs at least one row")
    has_clock = dataset.ts is not None
    reveal = dict(reveal or {})
    if reveal and not has_clock:
        raise ValueError("a reveal lag needs a dataset with timestamps")
    if set(reveal) - set(targets):
        raise KeyError(
            f"reveal names non-targets: {sorted(set(reveal) - set(targets))}"
        )
    labels = {t: _lag(reveal[t]) if t in reveal else None for t in targets}
    window = [str(d) for d in dates.span(dataset)] if has_clock else None
    recipe = {"source": source, "params": params or {}, "targets": labels}
    missing = [t for t in targets if t not in dataset.columns]
    if missing:
        raise KeyError(f"targets not in dataset: {missing}")
    text = [t for t in targets if not _numeric(dataset.columns[t].dtype)]
    if text:
        raise TypeError(
            f"targets {text} are not numeric; targets are masked with NaN, so encode "
            "labels as numbers"
        )
    wide = {n: a.shape for n, a in dataset.columns.items() if a.ndim != 1}
    if wide:
        raise Refused(
            f"columns of shape {wide}; a column is 1-D, one value per row, so split a "
            "2-D array into named columns"
        )
    sealed = _sealed(dataset, sealed_from)
    dataset_id = data_id(dataset, labels, sealed)
    if ledger.latest(Event.DATASET, dataset_id) is not None:
        return dataset_id
    sha = ledger.put_blob(rows_save(dataset))
    payload = {
        "source": source,
        "window": window,
        "recipe": identity.canonical(recipe),
        "rows": dataset.rows,
        "sealed_from": sealed,
        "blob": {"sha": sha, "format": formats.Format.PARQUET},
    }
    ledger.append(Event.DATASET, dataset_id, dataset_id, payload, id=dataset_id)
    return dataset_id


def _numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _sealed(dataset: Dataset, sealed_from: int | str | None) -> int | None:
    """The row the sealed tail starts at, inside the dataset, on the first row of its
    timestamp.
    """
    if sealed_from is None:
        return None
    row = sealed_from
    if not isinstance(row, (int, np.integer)):
        row = dataset.index_of(sealed_from)
    elif dataset.ts is not None and 0 < row < dataset.rows:
        row = dataset.index_of(dataset.ts[row])
    if not 0 < row < dataset.rows:
        raise Refused(
            f"sealed_from {sealed_from!r} is row {row} of {dataset.rows}; a sealed "
            "tail starts after the first row and before the last"
        )
    return int(row)


def _lag(reveal: datetime.timedelta | np.timedelta64 | int) -> float | dict[str, int]:
    """Seconds for a time lag, ``{"dates": n}`` for a count of later dates."""
    time = isinstance(reveal, (datetime.timedelta, np.timedelta64))
    if not time and not isinstance(reveal, (int, np.integer)):
        raise TypeError(
            f"a reveal lag is a timedelta or an int, not {type(reveal).__name__}"
        )
    if not time and reveal < 1:
        raise Refused(
            "an integer lag counts later dates and starts at 1; a same-date reveal is "
            "a timedelta"
        )
    lag = reveal / np.timedelta64(1, "s") if time else reveal
    if lag < 0:
        raise ValueError(f"a reveal lag cannot be negative: {reveal!r}")
    return lag if time else {"dates": lag}


def data_id(
    dataset: Dataset,
    targets: Mapping[str, Any] | None = None,
    sealed_from: int | None = None,
) -> str:
    """The content id of a dataset: its ``ts`` bytes, column names, dtypes and values
    (a string column by its strings), which columns are targets with their reveal lags
    in seconds or a count of dates, and the sealed tail's first row when it has one.
    """
    ts = None if dataset.ts is None else identity.bytes_hash(dataset.ts.tobytes())
    columns = {
        name: [str(a.dtype), identity.array_hash(a)]
        for name, a in dataset.columns.items()
    }
    sealed = {} if sealed_from is None else {"sealed_from": sealed_from}
    return identity.content_hash(
        {"ts": ts, "columns": columns, "targets": targets, **sealed}
    )


def resolve(ledger: Ledger, prefix: str) -> str:
    """The one recorded dataset id starting with ``prefix``; none or several are
    refused, naming the matches.
    """
    rows = ledger.sql("SELECT id FROM event_dataset ORDER BY seq")
    matches = [r["id"] for r in rows if r["id"].startswith(prefix)]
    if len(matches) != 1:
        hint = "pass a longer prefix: " + ", ".join(matches)
        raise Refused(
            f"dataset {prefix}: {len(matches)} matches in {ledger.root.resolve()}; "
            f"{hint if matches else 'lab ingest first'}"
        )
    return matches[0]


def load(ledger: Ledger, dataset_id: str) -> Dataset:
    """Load a recorded dataset's rows, resident."""
    event = ledger.latest(Event.DATASET, dataset_id)
    if event is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return rows_load(ledger.blobs / event["payload"]["blob"]["sha"])


# Bytes ================================================================================


def rows_save(dataset: Dataset) -> bytes:
    """A dataset as one Parquet file: ``ts`` as a nanosecond timestamp plus one column
    per array.
    """
    if TS in dataset.columns:
        raise ValueError(
            f"a dataset column cannot be named {TS!r}: ts is the clock's name"
        )
    stamps = {} if dataset.ts is None else {TS: dataset.ts}
    return formats.parquet_bytes(pl.DataFrame({**stamps, **dataset.columns}))


def rows_load(source: pathlib.Path | bytes) -> Dataset:
    """A stored dataset, read one column at a time so loading peaks near its size."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    names = list(pl.read_parquet_schema(source))
    columns = {
        n: pl.read_parquet(source, columns=[n])[n].to_numpy() for n in names if n != TS
    }
    ts = pl.read_parquet(source, columns=[TS])[TS].to_numpy() if TS in names else None
    return Dataset(columns, ts)


def from_frame(frame: pl.DataFrame, ts: str = TS) -> Dataset:
    """A dataset from a frame; a ``ts`` timestamp column is optional."""
    stamps = frame[ts].to_numpy() if ts in frame.columns else None
    columns = {name: frame[name].to_numpy() for name in frame.columns if name != ts}
    return Dataset(columns, stamps)
