"""Time-ordered frames and the resident session a pipeline reads ranges from.

Examples
--------
>>> import numpy as np
>>> ts = np.array(["2025-01-01T00:00", "2025-01-02T00:00"], dtype="datetime64[s]")
>>> s = Session(Frame(ts, {"x": np.array([1.0, 2.0])}))
>>> s.index_of("2025-01-02")
1
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np

Range = tuple[int, int]


@dataclasses.dataclass
class Frame:
    """Columns over a sorted timestamp axis."""

    ts: np.ndarray
    columns: dict[str, np.ndarray]

    def __post_init__(self):
        if np.any(np.diff(self.ts.astype("datetime64[s]").astype(np.int64)) < 0):
            raise ValueError("frame timestamps must be sorted")

    @property
    def rows(self) -> int:
        return int(self.ts.shape[0])

    @property
    def start(self) -> datetime.date:
        return self.ts[0].astype("datetime64[D]").astype(datetime.date)

    @property
    def end_exclusive(self) -> datetime.date:
        return self.ts[-1].astype("datetime64[D]").astype(datetime.date) + datetime.timedelta(
            days=1
        )


class Session:
    """A frame held resident; pipelines read row ranges, never the whole."""

    def __init__(self, frame: Frame):
        self.frame = frame
        self._seconds = frame.ts.astype("datetime64[s]").astype(np.int64)

    def __repr__(self) -> str:
        frame = self.frame
        return f"Session(rows={frame.rows}, start={frame.start}, end={frame.end_exclusive})"

    def index_of(
        self, when: str | datetime.date | datetime.datetime, offset_seconds: int = 0
    ) -> int:
        """First row at or after ``when`` shifted by ``offset_seconds``."""
        moment = np.datetime64(as_datetime(when), "s").astype(np.int64) + offset_seconds
        return int(np.searchsorted(self._seconds, moment, side="left"))

    def matrix(self, rng: Range, cols: tuple[str, ...]) -> np.ndarray:
        """Column-stacked features over a range, shape (rows, len(cols))."""
        return np.column_stack([self.frame.columns[c][rng[0] : rng[1]] for c in cols])

    def column(self, name: str, rng: Range) -> np.ndarray:
        return self.frame.columns[name][rng[0] : rng[1]]


# Date helpers =====================================================================================


def as_date(when: str | datetime.date | datetime.datetime) -> datetime.date:
    """Coerce a date-like value to a date."""
    if isinstance(when, datetime.datetime):
        return when.date()
    if isinstance(when, datetime.date):
        return when
    return datetime.date.fromisoformat(when[:10])


def as_datetime(when: str | datetime.date | datetime.datetime) -> datetime.datetime:
    if isinstance(when, datetime.datetime):
        return when
    date = as_date(when)
    return datetime.datetime(date.year, date.month, date.day)


def add_months(date: datetime.date, months: int) -> datetime.date:
    """Shift a date by whole months, clamping the day to the month's length."""
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    last_day = [31, 29 if _leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return datetime.date(year, month, min(date.day, last_day))


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
