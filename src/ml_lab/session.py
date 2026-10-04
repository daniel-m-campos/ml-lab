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

import numpy as np

Range = tuple[int, int]
Rows = Range | tuple[Range, ...]


class Session:
    """Columns over rows in a fixed order, with date lookups when ``ts`` is given."""

    def __init__(self, columns: dict[str, np.ndarray], ts: np.ndarray | None = None):
        self.columns = columns
        self.ts = None if ts is None else ts.astype("datetime64[ns]")
        if self.ts is not None and np.any(self.ts[1:] < self.ts[:-1]):
            raise ValueError("timestamps must be sorted")

    def __repr__(self) -> str:
        return f"Session(rows={self.rows}, clock={self.ts is not None})"

    @property
    def rows(self) -> int:
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
        return view

    def masked(self, columns: tuple[str, ...], start: int) -> Session:
        """A view with ``columns`` set to NaN from ``start`` on: the targets inside a
        prediction window, hidden from predict and postprocess.
        """
        view = self.upto(self.rows)
        for name in columns:
            values = view.columns[name].astype(np.float64, copy=True)
            values[start:] = np.nan
            view.columns[name] = values
        return view

    def frozen(self, start: int) -> Session:
        """A view whose rows from ``start`` on repeat row ``start - 1``: a future that
        never moves, so a step that reads it predicts differently before ``start``.
        """
        view = self.upto(self.rows)
        for name, values in view.columns.items():
            values = values.copy()
            values[start:] = values[start - 1]
            view.columns[name] = values
        return view

    def matrix(self, rows: Rows, cols: tuple[str, ...]) -> np.ndarray:
        """Column-stacked features over a range or segments, shape (rows, len(cols))."""
        return np.column_stack([self.column(c, rows) for c in cols])

    def column(self, name: str, rows: Rows) -> np.ndarray:
        values = self.columns[name]
        end = max(hi for _, hi in segments(rows))
        if end > self.rows:
            raise ValueError(
                f"rows up to {end} asked, {self.rows} visible before the cutoff"
            )
        return np.concatenate([values[lo:hi] for lo, hi in segments(rows)])

    def _clock(self) -> np.ndarray:
        if self.ts is None:
            raise ValueError("session has no timestamps; use a row-based split")
        return self.ts


def segments(rows: Rows) -> tuple[Range, ...]:
    """A range or a tuple of ranges as a tuple of ranges."""
    return (rows,) if isinstance(rows[0], (int, np.integer)) else rows
