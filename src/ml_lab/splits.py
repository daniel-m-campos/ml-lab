"""Splits: how a dataset becomes folds. Row-based splits need no clock; with one,
every cut lands on the first row of its timestamp, and with a ``group`` column first on
the first row of its group. An embargo edge is a count of rows from a cut and does not
snap, so ``embargo_rows`` can end a train segment inside a timestamp; the calendar
walk-forward reads the dataset's ``ts`` and embargoes whole timestamps.

A fold, ``Fold(label, train, test)``, trains on the ``train`` segments and scores the
``test`` range; each label appears once. A split is a frozen dataclass with
``folds(dataset) -> list[Fold]``, so it hashes into the evaluation id like any
declaration, and a project can declare its own next to its functions. A ``group``
that recurs after another group is refused, naming the row.

Examples
--------
>>> WalkForward(1000, 500, test_rows=500).folds(dataset)[1].test  # doctest: +SKIP
(1500, 2000)
>>> BlockedKFold(n_splits=5, embargo_rows=600).folds(dataset)[2].label  # doctest: +SKIP
'block 2'
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np

from ml_lab import dates
from ml_lab.dataset import Dataset, Range, Segments
from ml_lab.identity import Refused


@dataclasses.dataclass(frozen=True)
class Fold:
    label: str
    train: Segments
    test: Range


@dataclasses.dataclass(frozen=True)
class WalkForward:
    """Cutoffs every ``step_rows`` from ``first_cutoff_rows``; train on every row before
    each, or on the last ``max_train_rows`` of them, and score the ``horizon``-th block
    of ``test_rows`` after it, so ``horizon=1`` scores the ``test_rows`` right after the
    cutoff. ``embargo_rows`` cuts the train side: it drops the rows just before the
    cutoff from train.
    """

    first_cutoff_rows: int
    step_rows: int
    test_rows: int
    embargo_rows: int = 0
    horizon: int = 1
    max_train_rows: int | None = None
    min_folds: int = 3

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(
            step_rows=(self.step_rows, 1),
            test_rows=(self.test_rows, 1),
            embargo_rows=(self.embargo_rows, 0),
            horizon=(self.horizon, 1),
            max_train_rows=(
                1 if self.max_train_rows is None else self.max_train_rows,
                1,
            ),
        )
        out: list[Fold] = []
        cutoff = self.first_cutoff_rows
        while cutoff + self.horizon * self.test_rows <= dataset.rows:
            end = _snap(dataset, max(cutoff - self.embargo_rows, 0))
            start = (
                0 if self.max_train_rows is None else max(end - self.max_train_rows, 0)
            )
            last = cutoff + self.horizon * self.test_rows
            test = (_snap(dataset, last - self.test_rows), _snap(dataset, last))
            out.append(Fold(f"row {cutoff}", ((start, end),), test))
            cutoff += self.step_rows
        return _at_least(out, self.min_folds)


@dataclasses.dataclass(frozen=True)
class CalendarWalkForward:
    """Cutoffs on the dataset clock every ``step`` units, a unit being a calendar month
    or a day the clock has rows on; train on everything before each cutoff minus
    ``embargo_timestamps`` distinct timestamps, which cuts the train side, and score the
    ``horizon``-th run of ``test_units`` units after it, so ``horizon=1`` scores the
    ``test_units`` units right after the cutoff. A validation range ends on a unit
    boundary at or before ``end`` (the day after the data when ``end`` is not given);
    the boundary after the last day is ``end`` itself, so the last trading day is
    scored. Folds that reach past ``end`` are dropped, never cut, so a test period at
    the end of the data stays unscored.
    """

    first_cutoff: str
    unit: str = "month"
    step: int = 1
    test_units: int = 1
    horizon: int = 1
    embargo_timestamps: int = 0
    end: str | None = None
    min_folds: int = 3

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(
            step=(self.step, 1),
            test_units=(self.test_units, 1),
            embargo_timestamps=(self.embargo_timestamps, 0),
            horizon=(self.horizon, 1),
        )
        stop = dates.as_date(self.end) if self.end else dates.span(dataset)[1]
        at = self._boundaries(dataset, stop)
        reach = self.horizon * self.test_units
        out: list[Fold] = []
        for c in range(0, len(at) - reach, self.step):
            cut = dataset.index_of(at[c])
            for _ in range(self.embargo_timestamps):
                if cut:
                    cut = int(np.searchsorted(dataset.ts, dataset.ts[cut - 1], "left"))
            test = (
                dataset.index_of(at[c + reach - self.test_units]),
                dataset.index_of(at[c + reach]),
            )
            out.append(Fold(str(at[c]), ((0, cut),), test))
        why = (
            f"{max(len(at) - 1, 0)} {self.unit}s from the first cutoff "
            f"{self.first_cutoff} to {stop}, {reach} needed per fold"
        )
        if self.unit == "month" and not self.end:
            first = dates.add_months(dates.as_date(self.first_cutoff), reach)
            why += (
                f"; a month is whole or dropped, so set end={first} or later to score "
                "a validation range that runs to the data's end, or use unit='day'"
            )
        return _at_least(out, self.min_folds, why)

    def _boundaries(self, dataset: Dataset, stop: datetime.date) -> list[datetime.date]:
        """Unit boundaries from the first cutoff to ``stop`` inclusive."""
        first = dates.as_date(self.first_cutoff)
        if self.unit == "month":
            out: list[datetime.date] = []
            while (boundary := dates.add_months(first, len(out))) <= stop:
                out.append(boundary)
            return out
        if self.unit != "day":
            raise Refused(f"CalendarWalkForward unit {self.unit!r}: month or day")
        days = np.unique(dataset._clock().astype("datetime64[D]")).astype(datetime.date)
        return [d for d in days if first <= d < stop] + [stop]


@dataclasses.dataclass(frozen=True)
class BlockedKFold:
    """``n_splits`` contiguous blocks over the first ``train_size`` of rows, the rows
    ``Holdout(train_size)`` trains on; each is scored once with the rest as train.
    ``embargo_rows`` cuts the train side: it drops that many train rows on either side
    of the scored block. With ``group``, a column whose groups are contiguous, every
    cut moves to the first row of its group, so no group is split.
    """

    n_splits: int
    embargo_rows: int = 0
    train_size: float = 1.0
    group: str | None = None

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(n_splits=(self.n_splits, 2), embargo_rows=(self.embargo_rows, 0))
        starts = _group_starts(dataset, self.group)
        n = _snap(dataset, round(dataset.rows * self.train_size), starts)
        bounds = [
            _snap(dataset, round(i * n / self.n_splits), starts)
            for i in range(self.n_splits + 1)
        ]
        out = []
        for i in range(self.n_splits):
            lo, hi = bounds[i], bounds[i + 1]
            train = tuple(
                seg
                for seg in ((0, lo - self.embargo_rows), (hi + self.embargo_rows, n))
                if seg[0] < seg[1]
            )
            out.append(Fold(f"block {i}", train, (lo, hi)))
        return _at_least(out, 2)


@dataclasses.dataclass(frozen=True)
class Holdout:
    """One fold: the first ``train_size`` of rows train and the rest is scored.
    ``embargo_rows`` cuts the test side: it drops that many rows after the cut from
    the scored range. With ``group``, a column whose groups are contiguous, the cut
    moves to the first row of its group.
    """

    train_size: float = 0.7
    embargo_rows: int = 0
    group: str | None = None

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(embargo_rows=(self.embargo_rows, 0))
        n = dataset.rows
        cut = _snap(
            dataset, round(n * self.train_size), _group_starts(dataset, self.group)
        )
        test = (min(cut + self.embargo_rows, n), n)
        return _at_least([Fold("holdout", ((0, cut),), test)], 1)


def _snap(dataset: Dataset, row: int, groups: np.ndarray | None = None) -> int:
    """The first row of ``row``'s group, when ``groups`` holds the groups' first rows,
    then of its timestamp, so a cut never splits a group or a cross-section.
    """
    if not 0 < row < dataset.rows:
        return row
    if groups is not None:
        row = int(groups[np.searchsorted(groups, row, "right") - 1])
    if dataset.ts is None:
        return row
    return int(np.searchsorted(dataset.ts, dataset.ts[row], "left"))


def _group_starts(dataset: Dataset, group: str | None) -> np.ndarray | None:
    """The first row of each run of equal values in the ``group`` column; a group that
    recurs after another is refused, naming the row.
    """
    if group is None:
        return None
    values = dataset.column(group)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    first = np.unique(values[starts], return_index=True)[1]
    if len(first) < len(starts):
        again = starts[np.setdiff1d(np.arange(len(starts)), first)[0]]
        raise Refused(
            f"group {group}={values[again]} recurs at row {again} after another "
            "group; sort the rows so each group is contiguous"
        )
    return starts


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
    empty = [f.label for f in folds if f.test[0] >= f.test[1]]
    if empty:
        raise Refused(
            f"split yields empty validation ranges: {empty}"
            + (f"; {why}" if why else "")
        )
    bare = [f.label for f in folds if all(lo >= hi for lo, hi in f.train)]
    if bare:
        raise Refused(f"split yields folds with no train rows: {bare}; start later")
    twice = sorted(
        {f.label for f in folds if [g.label for g in folds].count(f.label) > 1}
    )
    if twice:
        raise Refused(f"split yields fold labels {twice} twice; label each fold once")
    return folds
