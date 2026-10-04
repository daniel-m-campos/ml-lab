"""A time-ordered table held resident; pipelines read row ranges, never the whole.

Examples
--------
>>> import numpy as np
>>> ts = np.array(["2025-01-01T00:00", "2025-01-02T00:00"], dtype="datetime64[s]")
>>> s = Session(ts, {"x": np.array([1.0, 2.0])})
>>> s.index_of("2025-01-02")
1
"""

from __future__ import annotations

import calendar
import datetime

import numpy as np

Range = tuple[int, int]


class Session:
    """Columns over a sorted timestamp axis, with range lookups by date."""

    def __init__(self, ts: np.ndarray, columns: dict[str, np.ndarray]):
        self.ts = ts.astype("datetime64[s]")
        self.columns = columns
        self._seconds = self.ts.astype(np.int64)
        if np.any(np.diff(self._seconds) < 0):
            raise ValueError("timestamps must be sorted")

    def __repr__(self) -> str:
        return f"Session(rows={self.rows}, start={self.start}, end={self.end_exclusive})"

    @property
    def rows(self) -> int:
        return int(self.ts.shape[0])

    @property
    def start(self) -> datetime.date:
        return self.date_at(0)

    @property
    def end_exclusive(self) -> datetime.date:
        return self.date_at(-1) + datetime.timedelta(days=1)

    def index_of(
        self, when: str | datetime.date | datetime.datetime, offset_seconds: int = 0
    ) -> int:
        """First row at or after ``when`` shifted by ``offset_seconds``."""
        midnight = datetime.datetime.combine(as_date(when), datetime.time())
        moment = np.datetime64(midnight, "s").astype(np.int64) + offset_seconds
        return int(np.searchsorted(self._seconds, moment, side="left"))

    def date_at(self, row: int) -> datetime.date:
        return self.ts[row].astype("datetime64[D]").astype(datetime.date)

    def matrix(self, rng: Range, cols: tuple[str, ...]) -> np.ndarray:
        """Column-stacked features over a range, shape (rows, len(cols))."""
        return np.column_stack([self.columns[c][rng[0] : rng[1]] for c in cols])

    def column(self, name: str, rng: Range) -> np.ndarray:
        return self.columns[name][rng[0] : rng[1]]


# Date helpers =====================================================================================


def as_date(when: str | datetime.date | datetime.datetime) -> datetime.date:
    """Coerce a date-like value to a date."""
    if isinstance(when, datetime.datetime):
        return when.date()
    if isinstance(when, datetime.date):
        return when
    return datetime.date.fromisoformat(when[:10])


def add_months(date: datetime.date, months: int) -> datetime.date:
    """Shift a date by whole months, clamping the day to the month's length."""
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    return datetime.date(year, month, min(date.day, calendar.monthrange(year, month)[1]))
