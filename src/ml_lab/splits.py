"""Splits: how a session becomes folds. Row-based splits need no clock; the calendar
walk-forward reads the session's ``ts``.

A fold trains on contiguous segments and scores one or more named windows. A splitter is
a frozen dataclass, so it hashes into the evaluation id like any declaration, and a
project can declare its own next to its steps.

Examples
--------
>>> WalkForward(first_cutoff_rows=1000, step_rows=500, window_rows=500).horizons
()
>>> BlockedKFold(k=5, embargo_rows=600).folds(session)[2].label  # doctest: +SKIP
'block 2'
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np

from ml_lab import dates
from ml_lab.identity import Refused
from ml_lab.session import Range, Session

Segments = tuple[Range, ...]


@dataclasses.dataclass(frozen=True)
class Fold:
    label: str
    train: Segments
    windows: dict[str, Range]


@dataclasses.dataclass(frozen=True)
class WalkForward:
    """Cutoffs every ``step_rows`` from ``first_cutoff_rows``; train on every row before
    each, minus ``embargo_rows``, and score ``window_rows`` after it. Without
    ``horizons`` the window is ``test``; with them, horizon ``h`` is the ``h``-th
    window after the cutoff, named ``str(h)``.
    """

    first_cutoff_rows: int
    step_rows: int
    window_rows: int
    horizons: tuple[int, ...] = ()
    embargo_rows: int = 0
    min_folds: int = 3

    def folds(self, session: Session) -> list[Fold]:
        names = self.horizons or (1,)
        _require(
            step_rows=(self.step_rows, 1),
            window_rows=(self.window_rows, 1),
            embargo_rows=(self.embargo_rows, 0),
            **{f"horizon {h}": (h, 1) for h in names},
        )
        out: list[Fold] = []
        cutoff = self.first_cutoff_rows
        while cutoff + max(names) * self.window_rows <= session.rows:
            train = ((0, max(cutoff - self.embargo_rows, 0)),)
            windows = {
                str(h) if self.horizons else "test": (
                    cutoff + (h - 1) * self.window_rows,
                    cutoff + h * self.window_rows,
                )
                for h in names
            }
            out.append(Fold(f"row {cutoff}", train, windows))
            cutoff += self.step_rows
        return _at_least(out, self.min_folds)


@dataclasses.dataclass(frozen=True)
class CalendarWalkForward:
    """Cutoffs on the session clock every ``every`` units, a unit being a calendar
    month or a day the clock has rows on; train on everything before each cutoff
    minus ``embargo_timestamps`` distinct timestamps, score the ``window`` units after
    it. Horizon ``h`` is the ``h``-th window after the cutoff, named ``str(h)``. A
    window ends on a unit boundary at or before ``end`` (the day after the data when
    ``end`` is not given); the boundary after the last day is ``end`` itself, so the
    last trading day is scored. Folds that reach past ``end`` are dropped, never cut,
    so a test period at the end of the data stays unscored.
    """

    first_cutoff: str
    unit: str = "month"
    every: int = 1
    window: int = 1
    horizons: tuple[int, ...] = (1, 2, 3)
    embargo_timestamps: int = 0
    end: str | None = None
    min_folds: int = 3

    def folds(self, session: Session) -> list[Fold]:
        _require(
            every=(self.every, 1),
            window=(self.window, 1),
            embargo_timestamps=(self.embargo_timestamps, 0),
            **{f"horizon {h}": (h, 1) for h in self.horizons},
        )
        stop = dates.as_date(self.end) if self.end else dates.span(session)[1]
        at = self._boundaries(session, stop)
        reach = max(self.horizons) * self.window
        out: list[Fold] = []
        for c in range(0, len(at) - reach, self.every):
            cut = session.index_of(at[c])
            for _ in range(self.embargo_timestamps):
                if cut:
                    cut = int(np.searchsorted(session.ts, session.ts[cut - 1], "left"))
            windows = {
                str(h): (
                    session.index_of(at[c + (h - 1) * self.window]),
                    session.index_of(at[c + h * self.window]),
                )
                for h in self.horizons
            }
            out.append(Fold(str(at[c]), ((0, cut),), windows))
        why = (
            f"{max(len(at) - 1, 0)} {self.unit}s from the first cutoff "
            f"{self.first_cutoff} to {stop}, {reach} needed per fold"
        )
        if self.unit == "month" and not self.end:
            first = dates.add_months(dates.as_date(self.first_cutoff), reach)
            why += (
                f"; a month is whole or dropped, so set end={first} or later to score "
                "a window that runs to the data's end, or use unit='day'"
            )
        return _at_least(out, self.min_folds, why)

    def _boundaries(self, session: Session, stop: datetime.date) -> list[datetime.date]:
        """Unit boundaries from the first cutoff to ``stop`` inclusive."""
        first = dates.as_date(self.first_cutoff)
        if self.unit == "month":
            out: list[datetime.date] = []
            while (boundary := dates.add_months(first, len(out))) <= stop:
                out.append(boundary)
            return out
        if self.unit != "day":
            raise Refused(f"CalendarWalkForward unit {self.unit!r}: month or day")
        days = np.unique(session._clock().astype("datetime64[D]")).astype(datetime.date)
        return [d for d in days if first <= d < stop] + [stop]


@dataclasses.dataclass(frozen=True)
class BlockedKFold:
    """``k`` contiguous blocks over the first ``train_fraction`` of rows, the rows
    ``Holdout(train_fraction)`` trains on; each is scored once as ``test`` with the
    rest as train, minus ``embargo_rows`` on either side of it.
    """

    k: int
    embargo_rows: int = 0
    train_fraction: float = 1.0

    def folds(self, session: Session) -> list[Fold]:
        _require(k=(self.k, 2), embargo_rows=(self.embargo_rows, 0))
        n = round(session.rows * self.train_fraction)
        bounds = [round(i * n / self.k) for i in range(self.k + 1)]
        out = []
        for i in range(self.k):
            lo, hi = bounds[i], bounds[i + 1]
            train = tuple(
                seg
                for seg in ((0, lo - self.embargo_rows), (hi + self.embargo_rows, n))
                if seg[0] < seg[1]
            )
            out.append(Fold(f"block {i}", train, {"test": (lo, hi)}))
        return _at_least(out, 2)


@dataclasses.dataclass(frozen=True)
class Holdout:
    """One fold: the first ``train_fraction`` of rows train, the rest after
    ``embargo_rows`` is scored as ``test``.
    """

    train_fraction: float = 0.7
    embargo_rows: int = 0

    def folds(self, session: Session) -> list[Fold]:
        _require(embargo_rows=(self.embargo_rows, 0))
        n = session.rows
        cut = round(n * self.train_fraction)
        test = (min(cut + self.embargo_rows, n), n)
        return _at_least([Fold("holdout", ((0, cut),), {"test": test})], 1)


def _require(**fields: tuple[int, int]):
    """Refuse a split field below its floor, given as ``name=(value, floor)``."""
    low = [f"{k}={v}" for k, (v, floor) in fields.items() if v < floor]
    if low:
        raise Refused(f"split fields out of range: {', '.join(low)}")


def _at_least(folds: list[Fold], minimum: int, why: str = "") -> list[Fold]:
    if len(folds) < max(minimum, 1):
        raise Refused(
            f"split yields {len(folds)} folds, at least {minimum} needed"
            + (f": {why}" if why else "")
        )
    empty = [
        f"{f.label} {n}" for f in folds for n, (lo, hi) in f.windows.items() if lo >= hi
    ]
    if empty:
        raise Refused(
            f"split yields empty windows: {empty}" + (f"; {why}" if why else "")
        )
    bare = [f.label for f in folds if all(lo >= hi for lo, hi in f.train)]
    if bare:
        raise Refused(f"split yields folds with no train rows: {bare}; start later")
    return folds
