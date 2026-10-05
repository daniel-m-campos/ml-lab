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
    """Columns over rows in a fixed order, with date lookups when ``ts`` is given.
    ``feature_columns`` names the columns a pipeline's features step added, in the
    step's order; ``()`` without one.
    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        ts: np.ndarray | None = None,
        feature_columns: tuple[str, ...] = (),
    ):
        self.columns = columns
        self.feature_columns = feature_columns
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
        view.feature_columns = self.feature_columns
        return view

    def masked(self, reveal: dict[str, np.timedelta64 | None], at: int) -> Session:
        """A view whose target columns are NaN for every label not known at row
        ``at``: a label with lag ``L`` is known from ``ts + L``, one without a lag
        only after its own timestamp; without a clock, rows from ``at`` on.
        """
        view = self.upto(self.rows)
        for name, lag in reveal.items():
            values = view.columns[name].astype(np.float64, copy=True)
            if self.ts is None:
                values[at:] = np.nan
            elif lag is None:
                values[self.ts >= self.ts[at]] = np.nan
            else:
                values[self.ts + lag > self.ts[at]] = np.nan
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
