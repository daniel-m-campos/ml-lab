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
from collections.abc import Callable

import numpy as np

from ml_lab import dates
from ml_lab.ledger import Refused
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
    it. Horizon ``h`` is the ``h``-th window after the cutoff, named ``str(h)``. Folds
    whose last window reaches past ``end`` are dropped, so a test period at the end of
    the data stays unscored.
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
        at = self._boundaries(session)
        reach = max(self.horizons) * self.window
        stop = dates.as_date(self.end) if self.end else dates.span(session)[1]
        out: list[Fold] = []
        c = 0
        while at(c + reach) <= stop:
            cut = session.index_of(at(c))
            for _ in range(self.embargo_timestamps):
                if cut:
                    cut = int(np.searchsorted(session.ts, session.ts[cut - 1], "left"))
            windows = {
                str(h): (
                    session.index_of(at(c + (h - 1) * self.window)),
                    session.index_of(at(c + h * self.window)),
                )
                for h in self.horizons
            }
            out.append(Fold(str(at(c)), ((0, cut),), windows))
            c += self.every
        why = (
            f"the first cutoff {at(0)} plus {reach} {self.unit}s reaches {at(reach)}, "
            f"past {stop}"
        )
        return _at_least(out, self.min_folds, why)

    def _boundaries(self, session: Session) -> Callable[[int], datetime.date]:
        """The n-th boundary date after the first cutoff."""
        first = dates.as_date(self.first_cutoff)
        if self.unit == "month":
            return lambda n: dates.add_months(first, n)
        if self.unit != "day":
            raise Refused(f"CalendarWalkForward unit {self.unit!r}: month or day")
        days = np.unique(session._clock().astype("datetime64[D]"))
        days = days[days >= np.datetime64(first)].astype(datetime.date)

        def at(n: int) -> datetime.date:
            return days[n] if n < len(days) else datetime.date.max

        return at


@dataclasses.dataclass(frozen=True)
class BlockedKFold:
    """``k`` contiguous blocks; each is scored once as ``test`` with the rest as train,
    minus ``embargo_rows`` on either side of it.
    """

    k: int
    embargo_rows: int = 0

    def folds(self, session: Session) -> list[Fold]:
        n = session.rows
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
        n = session.rows
        cut = round(n * self.train_fraction)
        test = (min(cut + self.embargo_rows, n), n)
        return _at_least([Fold("holdout", ((0, cut),), {"test": test})], 1)


def _at_least(folds: list[Fold], minimum: int, why: str = "") -> list[Fold]:
    if len(folds) < minimum:
        raise Refused(
            f"split yields {len(folds)} folds, at least {minimum} needed"
            + (f": {why}" if why else "")
        )
    if any(lo >= hi for f in folds for lo, hi in f.windows.values()):
        raise Refused("split yields an empty window")
    return folds
