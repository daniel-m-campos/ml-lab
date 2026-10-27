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
>>> inner, valid = inner_split(dataset, train, 0.1)  # inside fit  # doctest: +SKIP
>>> oof_folds(dataset, train, "block 2")[0].label  # doctest: +SKIP
'block 2 oof 0'
"""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np

from ml_lab import dates
from ml_lab.dataset import Dataset, FitView, Range, Segments, known_before
from ml_lab.identity import Refused

OOF_SPLITS = 5


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
            end = purge(dataset, cutoff, embargo_rows=self.embargo_rows)
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
    or a day the clock has rows on.

    Each fold trains on everything before its cutoff, or with ``train_units`` on the
    last that many units before it, minus ``embargo_timestamps`` distinct timestamps
    from the train side, and scores the ``horizon``-th run of ``test_units`` units
    after it. A validation range ends on a unit boundary at or before ``end``; the
    boundary after the last day is ``end`` itself, so the last trading day is
    scored. Folds that reach past ``end`` are dropped, never cut, so a test period
    at the end of the data stays unscored.

    Parameters
    ----------
    first_cutoff : str
        The first cutoff date.
    unit : str
        ``"month"`` or ``"day"``.
    step : int
        Units between cutoffs.
    test_units : int
        Units each fold scores.
    horizon : int
        Which run of ``test_units`` after the cutoff is scored; 1 is the run right
        after it.
    embargo_timestamps : int
        Distinct timestamps dropped from the train end before each cutoff.
    end : str, optional
        The last boundary; the day after the data when not given.
    min_folds : int
        The fewest folds the split may yield.
    train_units : int, optional
        A rolling window: the units of train before each cutoff; None trains from
        the first row.
    """

    first_cutoff: str
    unit: str = "month"
    step: int = 1
    test_units: int = 1
    horizon: int = 1
    embargo_timestamps: int = 0
    end: str | None = None
    min_folds: int = 3
    train_units: int | None = None

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(
            step=(self.step, 1),
            test_units=(self.test_units, 1),
            embargo_timestamps=(self.embargo_timestamps, 0),
            horizon=(self.horizon, 1),
            train_units=(1 if self.train_units is None else self.train_units, 1),
        )
        stop = dates.as_date(self.end) if self.end else dates.span(dataset)[1]
        at = self._boundaries(dataset, stop)
        reach = self.horizon * self.test_units
        out: list[Fold] = []
        for c in range(0, len(at) - reach, self.step):
            cut = purge(
                dataset,
                dataset.index_of(at[c]),
                embargo_timestamps=self.embargo_timestamps,
            )
            test = (
                dataset.index_of(at[c + reach - self.test_units]),
                dataset.index_of(at[c + reach]),
            )
            units = self.train_units
            start = 0 if units is None else self._window(dataset, at[c], units)
            out.append(Fold(str(at[c]), ((start, cut),), test))
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

    def _window(self, dataset: Dataset, cutoff: datetime.date, units: int) -> int:
        """The first row of the rolling window: ``units`` units before the cutoff,
        months by the calendar, days by the clock's days with rows.
        """
        if self.unit == "month":
            return dataset.index_of(dates.add_months(cutoff, -units))
        days = np.unique(dataset._clock().astype("datetime64[D]"))
        at = int(np.searchsorted(days, np.datetime64(cutoff, "D")))
        return dataset.index_of(str(days[max(at - units, 0)]))

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
    ``train_size`` is a fraction of the rows, or an int, that many rows, so a sealed
    tail is named by its first row and the cut cannot round past it. ``embargo_rows``
    cuts the train side: it drops that many train rows on either side of the scored
    block. With ``group``, a column whose groups are contiguous, every cut moves to
    the first row of its group, so no group is split. The dataset's auxiliary rows
    are train in every fold and never scored: the blocks tile the rows after them,
    so they match a split of the judged rows alone row for row.
    """

    n_splits: int
    embargo_rows: int = 0
    train_size: int | float = 1.0
    group: str | None = None

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(n_splits=(self.n_splits, 2), embargo_rows=(self.embargo_rows, 0))
        starts = _group_starts(dataset, self.group)
        n = _snap(dataset, _train_rows(dataset, self.train_size), starts)
        head = dataset.auxiliary_rows
        if head >= n:
            raise Refused(
                f"train_size={self.train_size!r} ends the split at row {n}, inside the "
                f"dataset's {head} auxiliary rows, which are never scored; make it "
                f"larger than {head}"
            )
        bounds = [
            _snap(dataset, head + round(i * (n - head) / self.n_splits), starts)
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
    ``train_size`` is a fraction of the rows, or an int, that many rows, so a sealed
    tail is named by its first row. ``embargo_rows`` cuts the test side: it drops
    that many rows after the cut from the scored range. With ``group``, a column
    whose groups are contiguous, the cut moves to the first row of its group.
    """

    train_size: int | float = 0.7
    embargo_rows: int = 0
    group: str | None = None

    def folds(self, dataset: Dataset) -> list[Fold]:
        _require(embargo_rows=(self.embargo_rows, 0))
        n = dataset.rows
        cut = _snap(
            dataset,
            _train_rows(dataset, self.train_size),
            _group_starts(dataset, self.group),
        )
        test = (min(cut + self.embargo_rows, n), n)
        return _at_least([Fold("holdout", ((0, cut),), test)], 1)


def inner_split(
    dataset: FitView,
    train: Segments,
    validation: int | float,
    embargo_timestamps: int | None = None,
) -> tuple[Segments, Range]:
    """Cut a fold's train segments, inside ``fit``, into an inner train and a
    validation range.

    For early stopping or any choice made on held-out rows. The validation range is
    the tail of the last train segment, its start snapped to the first row of its
    timestamp, or that whole segment when it holds no more rows than asked and
    earlier segments remain to train on, as the inner train of an out-of-fold block
    can; it never reaches into the dataset's auxiliary rows, which are judged by no
    evaluation. The inner train is the train segments before it, purged as the
    evaluation purges: the view's embargo, then every row whose label the view's
    reveal lags say is not known when the validation starts, so a round count chosen
    here reads no label the fold's own test would hide. A ``group`` column is not
    honoured, so a grouped split cuts its own.

    Parameters
    ----------
    dataset : FitView
        The view ``fit`` received; its embargo and reveal lags drive the purge.
    train : Segments
        The train segments to cut, normally ``fit``'s ``train`` argument.
    validation : int | float
        Rows in the validation range, or that fraction of the train rows when below
        1.
    embargo_timestamps : int, optional
        Overrides the view's ``embargo_timestamps`` alone.

    Returns
    -------
    tuple[Segments, Range]
        The inner train segments and the validation range.

    Raises
    ------
    Refused
        When ``dataset`` is not a ``FitView``, ``train`` is empty, ``validation`` is
        not a count or a fraction, or either side comes out empty.
    """
    if not isinstance(dataset, FitView):
        raise Refused(
            "inner_split reads the embargo and reveal lags from the view a fit "
            "receives; pass fit's dataset argument, a FitView"
        )
    rows = sum(hi - lo for lo, hi in train)
    if not train or rows == 0:
        raise Refused("inner_split needs train segments with rows")
    if isinstance(validation, float) and 0 < validation < 1:
        held = int(round(validation * rows))
    elif isinstance(validation, int) and not isinstance(validation, bool):
        held = validation
    else:
        raise Refused(
            f"validation {validation!r}: a count of rows, or a fraction in (0, 1)"
        )
    start, end = train[-1]
    head = dataset.auxiliary_rows
    start = max(start, head)
    at = _snap(dataset, max(end - held, start))
    if at >= end or (at <= start and len(train) == 1):
        where = f" after the dataset's {head} auxiliary rows" if head else ""
        raise Refused(
            f"validation of {held} rows leaves no inner train or no validation in the "
            f"last train segment {train[-1]}{where}; ask for fewer rows or a smaller "
            "fraction"
        )
    if embargo_timestamps is None:
        embargo_timestamps = dataset.embargo_timestamps
    cut = purge(dataset, at, dataset.embargo_rows, embargo_timestamps)
    for lag in dataset.reveal.values():
        cut = min(cut, known_before(dataset, at, lag))
    inner = tuple((lo, min(hi, cut)) for lo, hi in train if lo < cut)
    if not inner:
        raise Refused(
            f"the embargo and reveal lags purge every inner train row before "
            f"validation row {at}; ask for a smaller validation"
        )
    return inner, (at, end)


def oof_folds(dataset: FitView, train: Segments, label: str = "") -> list[Fold]:
    """Cut a fold's train segments into ``OOF_SPLITS`` contiguous blocks, each a fold
    that trains on the other train rows and predicts the block: the out-of-fold
    predictions a learned blend receives.

    The blocks tile the train rows after the dataset's auxiliary rows in order, cut
    over the segments as one row range, every inner bound snapped to the first row of
    its timestamp; the auxiliary rows stay in every block's train, as they do in
    every fold's. A block's train is purged on both sides as the evaluation purges:
    the view's embargo before and after the block, and before it every row whose
    label the view's reveal lags say is not known when the block starts. A block the
    segments split in two yields a fold per part, sharing the train. A ``group``
    column is not honoured.

    Parameters
    ----------
    dataset : FitView
        The view the blend's fit receives; its clock, embargo and reveal lags drive the
        cut and the purge.
    train : Segments
        The fold's train segments.
    label : str
        The fold's label; each block's fold is labelled ``"<label> oof <k>"``.

    Returns
    -------
    list[Fold]
        One fold per block part, in row order.

    Raises
    ------
    Refused
        When ``train`` holds fewer rows after the auxiliary rows than blocks, a block
        comes out empty, or the purge leaves a block no train row.
    """
    head = dataset.auxiliary_rows
    judged = tuple((max(lo, head), hi) for lo, hi in train if hi > head)
    rows = sum(hi - lo for lo, hi in judged)
    if not judged or rows < OOF_SPLITS:
        raise Refused(
            f"out-of-fold predictions cut the train rows after the dataset's {head} "
            f"auxiliary rows into {OOF_SPLITS} blocks, {rows} rows given"
        )
    bounds = [judged[0][0]]
    bounds += [
        _snap(dataset, _absolute(judged, round(i * rows / OOF_SPLITS)))
        for i in range(1, OOF_SPLITS)
    ]
    bounds.append(judged[-1][1])
    out = []
    for k in range(OOF_SPLITS):
        lo, hi = bounds[k], bounds[k + 1]
        blocks = [
            (max(lo, a), min(hi, b)) for a, b in judged if max(lo, a) < min(hi, b)
        ]
        if not blocks:
            raise Refused(
                f"block {k} of the out-of-fold cut is empty: the timestamps snap its "
                f"bounds {lo}:{hi} together"
            )
        before = purge(dataset, lo, dataset.embargo_rows, dataset.embargo_timestamps)
        for lag in dataset.reveal.values():
            before = min(before, known_before(dataset, lo, lag))
        after = _advance(dataset, hi + dataset.embargo_rows, dataset.embargo_timestamps)
        inner = tuple(
            seg
            for a, b in train
            for seg in ((a, min(b, before)), (max(a, after), b))
            if seg[0] < seg[1]
        )
        if not inner:
            raise Refused(
                f"the embargo and reveal lags purge every train row around block {k} "
                f"of the out-of-fold cut, rows {lo}:{hi}"
            )
        name = f"{label} oof {k}".strip()
        out.extend(Fold(name, inner, block) for block in blocks)
    return out


def purge(
    dataset: Dataset | FitView,
    cut: int,
    embargo_rows: int = 0,
    embargo_timestamps: int = 0,
) -> int:
    """The last train row before ``cut``, with the embargo applied.

    ``embargo_rows`` rows back, snapped to the first row of its timestamp, then
    ``embargo_timestamps`` distinct timestamps back (rows, without a clock). The
    walk-forward splits and ``inner_split`` cut the train side with this one rule.

    Parameters
    ----------
    dataset : Dataset | FitView
        The rows and clock.
    cut : int
        The row the train must end before.
    embargo_rows : int
        Rows dropped before ``cut``.
    embargo_timestamps : int
        Distinct timestamps dropped after the row embargo.

    Returns
    -------
    int
        The train end, a row at or before ``cut``.
    """
    cut = _snap(dataset, max(cut - embargo_rows, 0))
    for _ in range(embargo_timestamps):
        if cut and dataset.ts is not None:
            cut = int(np.searchsorted(dataset.ts, dataset.ts[cut - 1], "left"))
        elif cut:
            cut -= 1
    return cut


def _train_rows(dataset: Dataset, train_size: int | float) -> int:
    """The rows ``train_size`` names: a fraction of the dataset's rounded to a row,
    or an int as it is.
    """
    if isinstance(train_size, bool) or not isinstance(train_size, (int, float)):
        raise Refused(f"train_size={train_size!r}: a fraction, or an int count of rows")
    if isinstance(train_size, int):
        return train_size
    return round(dataset.rows * train_size)


def _absolute(train: Segments, offset: int) -> int:
    """The row ``offset`` rows into the train segments read as one range."""
    for lo, hi in train:
        if offset <= hi - lo:
            return lo + offset
        offset -= hi - lo
    return train[-1][1]


def _advance(dataset: Dataset | FitView, row: int, timestamps: int) -> int:
    """The first row ``timestamps`` distinct timestamps after ``row``'s; ``row`` itself
    without a clock.
    """
    for _ in range(timestamps):
        if dataset.ts is None or not 0 <= row < dataset.rows:
            break
        row = int(np.searchsorted(dataset.ts, dataset.ts[row], "right"))
    return row


def _snap(
    dataset: Dataset | FitView, row: int, groups: np.ndarray | None = None
) -> int:
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
        raise Refused(
            f"split yields folds with no train rows: {bare}; start later, or with a "
            "rolling window make it longer than the embargo"
        )
    twice = sorted(
        {f.label for f in folds if [g.label for g in folds].count(f.label) > 1}
    )
    if twice:
        raise Refused(f"split yields fold labels {twice} twice; label each fold once")
    return folds
