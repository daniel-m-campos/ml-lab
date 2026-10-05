"""A table held resident, read by row ranges, with a clock lookup when ``ts`` is given.

Examples
--------
>>> import numpy as np
>>> ts = np.array(["2025-01-01T00:00", "2025-01-02T00:00"], dtype="datetime64[s]")
>>> s = Session({"x": np.array([1.0, 2.0])}, ts)
>>> s.index_of("2025-01-02")
1
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import polars as pl

Range = tuple[int, int]
Rows = Range | tuple[Range, ...]
Lag = np.timedelta64 | int | None


class Session:
    """Columns over rows in a fixed order, with date lookups when ``ts`` is given.
    ``feature_columns`` names the columns a pipeline's features steps added, in step
    order; they are not in ``columns`` but read on demand from the Parquet file in
    ``stores`` through ``column`` and ``matrix``, so a column no step names costs no
    memory.
    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        ts: np.ndarray | None = None,
        feature_columns: tuple[str, ...] = (),
        stores: dict[str, pathlib.Path] | None = None,
    ):
        self.columns = columns
        self.feature_columns = feature_columns
        self.stores = stores or {}
        self.cut: int | None = None
        self.ts = None if ts is None else ts.astype("datetime64[ns]")
        if self.ts is not None and np.any(self.ts[1:] < self.ts[:-1]):
            raise ValueError("timestamps must be sorted")

    def __repr__(self) -> str:
        return f"Session(rows={self.rows}, clock={self.ts is not None})"

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

    def upto(self, row: int) -> Session:
        """The first ``row`` rows as a view: positions are unchanged, so ranges into
        the full session stay valid, and nothing after ``row`` can be read.
        """
        view = Session.__new__(Session)
        view.columns = {k: v[:row] for k, v in self.columns.items()}
        view.ts = None if self.ts is None else self.ts[:row]
        view.feature_columns = self.feature_columns
        view.stores = self.stores
        view.cut = min(row, self.rows)
        return view

    def masked(self, reveal: dict[str, Lag], at: int) -> Session:
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

    def boundary(self, row: int, lo: int = 0) -> int | None:
        """The first row of ``row``'s timestamp, or of the next one when that is
        ``lo``; None when no timestamp starts inside ``(lo, rows)``. Without a clock
        every row is its own timestamp. A probe cuts here so it never splits a
        cross-section.
        """
        if self.ts is None:
            return row if lo < row < self.rows else None
        first = int(np.searchsorted(self.ts, self.ts[row], "left"))
        if first > lo:
            return first
        after = int(np.searchsorted(self.ts, self.ts[row], "right"))
        return after if after < self.rows else None

    def matrix(self, rows: Rows, cols: tuple[str, ...]) -> np.ndarray:
        """Column-stacked features over a range or segments, shape (rows, len(cols)),
        filled one column at a time so the matrix is the only copy.
        """
        parts = segments(rows)
        first = self.column(cols[0], parts)
        dtype = np.result_type(first, *(self._dtype(c) for c in cols[1:]))
        out = np.empty((len(first), len(cols)), dtype)
        out[:, 0] = first
        for j, name in enumerate(cols[1:], 1):
            out[:, j] = self.column(name, parts)
        return out

    def column(self, name: str, rows: Rows) -> np.ndarray:
        end = max(hi for _, hi in segments(rows))
        if end > self.rows:
            raise ValueError(
                f"rows up to {end} asked, {self.rows} visible before the cutoff"
            )
        if name in self.columns:
            values = self.columns[name]
            return np.concatenate([values[lo:hi] for lo, hi in segments(rows)])
        if name not in self.stores:
            raise KeyError(
                f"{name!r} is not a column: dataset columns {sorted(self.columns)}, "
                f"feature columns {list(self.feature_columns)}"
            )
        scan = pl.scan_parquet(self.stores[name]).select(name)
        return np.concatenate(
            [
                scan.slice(lo, hi - lo).collect()[name].to_numpy()
                for lo, hi in segments(rows)
            ]
        )

    def _dtype(self, name: str) -> np.dtype:
        if name in self.columns:
            return self.columns[name].dtype
        schema = pl.read_parquet_schema(self.stores[name])
        return pl.Series([], dtype=schema[name]).to_numpy().dtype

    def _clock(self) -> np.ndarray:
        if self.ts is None:
            raise ValueError("session has no timestamps; use a row-based split")
        return self.ts


def segments(rows: Rows) -> tuple[Range, ...]:
    """A range or a tuple of ranges as a tuple of ranges."""
    return (rows,) if isinstance(rows[0], (int, np.integer)) else rows
