"""Calendar arithmetic for sessions with a clock: the calendar split, month lookbacks.

Examples
--------
>>> add_months(as_date("2025-01-31"), 1)
datetime.date(2025, 2, 28)
"""

from __future__ import annotations

import calendar
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forestry.session import Session


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
    return datetime.date(
        year, month, min(date.day, calendar.monthrange(year, month)[1])
    )


def span(session: Session) -> tuple[datetime.date, datetime.date]:
    """The session's first date and the day after its last."""
    return session.date_at(0), session.date_at(-1) + datetime.timedelta(days=1)
