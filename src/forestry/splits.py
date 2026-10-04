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

from forestry import dates
from forestry.ledger import Refused
from forestry.session import Range, Session

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
    """Monthly cutoffs on the session clock; train on everything before each, score the
    months after it. Horizon ``h`` is the ``h``-th block of ``eval_months`` after the
    cutoff, named ``str(h)``. Train ends ``embargo_seconds`` before the cutoff.
    """

    first_cutoff: str
    every_months: int = 1
    eval_months: int = 1
    horizons: tuple[int, ...] = (1, 2, 3)
    embargo_seconds: int = 0
    min_folds: int = 3

    def folds(self, session: Session) -> list[Fold]:
        reach = max(self.horizons) * self.eval_months
        end = dates.span(session)[1]
        out: list[Fold] = []
        cutoff = dates.as_date(self.first_cutoff)
        while dates.add_months(cutoff, reach) <= end:
            train = ((0, session.index_of(cutoff, -self.embargo_seconds)),)
            windows = {
                str(h): (
                    session.index_of(
                        dates.add_months(cutoff, (h - 1) * self.eval_months)
                    ),
                    session.index_of(dates.add_months(cutoff, h * self.eval_months)),
                )
                for h in self.horizons
            }
            out.append(Fold(str(cutoff), train, windows))
            cutoff = dates.add_months(cutoff, self.every_months)
        return _at_least(out, self.min_folds)


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


def _at_least(folds: list[Fold], minimum: int) -> list[Fold]:
    if len(folds) < minimum:
        raise Refused(f"split yields {len(folds)} folds, at least {minimum} needed")
    if any(lo >= hi for f in folds for lo, hi in f.windows.values()):
        raise Refused("split yields an empty window")
    return folds
