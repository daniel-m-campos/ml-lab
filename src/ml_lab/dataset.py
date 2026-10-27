"""Datasets: columns over rows in a fixed order, recorded once and loaded back.

A ``Dataset`` is a table held resident, read by row ranges through ``column(name,
rows)`` and ``matrix(cols, rows)``, with a clock lookup when ``ts`` is given;
``from_frame`` builds one from a polars frame. A recorded dataset is the rows of one
(source, params), cleaned by the caller before ``record``, with named target columns.
Its id is the data: the ``ts`` bytes, column names, dtypes and values, so a loader fix
that changes rows is a new dataset and an edit that changes nothing is not. The recipe
rides on the event as provenance; the bytes are Parquet.

The rows may have two boundaries, both recorded with them and in the id: an
auxiliary head, ``record(auxiliary_rows=n)``, the first rows, which a fit may learn
from and no evaluation scores, and a sealed tail, ``record(sealed_from=row)``, which
only a sealed evaluation scores. A fit reads the head as ``dataset.auxiliary_rows``
on its view.

Examples
--------
>>> import numpy as np
>>> ts = np.array(["2025-01-01T00:00", "2025-01-02T00:00"], dtype="datetime64[s]")
>>> d = Dataset({"x": np.array([1.0, 2.0])}, ts)
>>> d.index_of("2025-01-02")
1
>>> dataset_id = record(ledger, d, source="toy", targets=("x",))  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import json
import pathlib
import sys
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, cast

import numpy as np
import polars as pl

from ml_lab import dates, formats, identity
from ml_lab.formats import TS
from ml_lab.identity import Refused
from ml_lab.ledger import Event, Ledger, actor

Range = tuple[int, int]
Rows = Range | tuple[Range, ...]
Segments = tuple[Range, ...]
Lag = np.timedelta64 | int | None


class Dataset:
    """Columns over rows in a fixed order, with date lookups when ``ts`` is given.

    Parameters
    ----------
    columns : dict[str, np.ndarray]
        Equal-length 1-D arrays in a fixed order; a datetime array is held in
        nanoseconds.
    ts : np.ndarray, optional
        Sorted timestamps, one per row, several rows to a timestamp for a
        cross-section; None for a dataset without a clock.
    computed_columns : tuple[str, ...], optional
        The columns a pipeline's features functions added, in order. They are not in
        ``columns``: ``column`` and ``matrix`` read them on demand from ``stores``,
        so a column no function names costs no memory.
    stores : dict[str, pathlib.Path], optional
        The Parquet file each computed column is read from.
    reveal : Mapping[str, Lag], optional
        The targets' reveal lags, recorded with the data and set by ``load``.
    auxiliary_rows : int, optional
        The auxiliary head: that many leading rows a fit may learn from and no
        evaluation scores, recorded with the data and set by ``load``.

    Raises
    ------
    ValueError
        When ``ts`` holds NaT or is not sorted.

    Notes
    -----
    What a fold adds for one fit is not here: ``FitView`` carries it.
    """

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        ts: np.ndarray | None = None,
        computed_columns: tuple[str, ...] = (),
        stores: dict[str, pathlib.Path] | None = None,
        reveal: Mapping[str, Lag] | None = None,
        auxiliary_rows: int = 0,
    ):
        self.columns = {
            n: a.astype("datetime64[ns]", copy=False) if a.dtype.kind == "M" else a
            for n, a in columns.items()
        }
        self.computed_columns = computed_columns
        self.stores = stores or {}
        self.cut: int | None = None
        self.keep: np.ndarray | None = None
        self.reveal: dict[str, Lag] = dict(reveal or {})
        self.auxiliary_rows = auxiliary_rows
        self.ts = None if ts is None else ts.astype("datetime64[ns]")
        if self.ts is not None and np.isnat(self.ts).any():
            raise ValueError("timestamps must not contain NaT")
        if self.ts is not None and np.any(self.ts[1:] < self.ts[:-1]):
            raise ValueError("timestamps must be sorted")

    def __repr__(self) -> str:
        return f"Dataset(rows={self.rows}, clock={self.ts is not None})"

    @property
    def rows(self) -> int:
        if self.cut is not None:
            return self.cut
        return int(next(iter(self.columns.values())).shape[0])

    def index_of(self, when: str | datetime.date | datetime.datetime) -> int:
        """First row at or after ``when``; a date means its midnight."""
        return int(np.searchsorted(self._clock(), np.datetime64(when, "ns"), "left"))

    def date_at(self, row: int) -> datetime.date:
        return self._clock()[row].astype("datetime64[D]").astype(datetime.date)

    def upto(self, row: int) -> Dataset:
        """The first ``row`` rows as a view: positions are unchanged, so ranges into
        the full dataset stay valid, and nothing after ``row`` can be read.
        """
        view = Dataset.__new__(Dataset)
        if isinstance(self.columns, _Blanked):
            view.columns = self.columns.upto(row)
        else:
            view.columns = {k: v[:row] for k, v in self.columns.items()}
        view.ts = None if self.ts is None else self.ts[:row]
        view.computed_columns = self.computed_columns
        view.stores = self.stores
        view.cut = min(row, self.rows)
        view.keep = None if self.keep is None else self.keep[:row]
        view.reveal = self.reveal
        view.auxiliary_rows = self.auxiliary_rows
        return view

    def masked(self, reveal: dict[str, Lag], at: int) -> Dataset:
        """A view whose target columns are NaN for every label not known at row
        ``at``: a label with a time lag ``L`` is known from ``ts + L``, one with an
        integer lag ``n`` from the first row of the n-th later date that has rows, one
        without a lag only after its own timestamp; without a clock, rows from ``at``
        on.
        """
        view = self.upto(self.rows)
        for name, lag in reveal.items():
            values = view.columns[name].astype(np.float64, copy=True)
            values[known_before(self, at, lag) :] = np.nan
            view.columns[name] = values
        return view

    def train_view(self, train: Segments) -> Dataset:
        """The view a fit sees: every column blank outside ``train``.

        Computed columns included, so a fit reads no row of a range it is scored on. A
        blank is NaN, NaT or None by dtype; an integer or boolean column becomes float.
        A column is copied when first read, so a fit that reads ten columns of two
        hundred pays for ten.

        Parameters
        ----------
        train : Segments
            The train row segments to keep.

        Returns
        -------
        Dataset
            A view over the same rows; ``FitView`` wraps it for ``fit``.
        """
        view = self.upto(self.rows)
        view.keep = row_mask(self.rows, train)
        view.columns = _Blanked(view.columns, view.keep)
        return view

    def boundary(self, row: int, lo: int = 0) -> int | None:
        """The first row of ``row``'s timestamp, or of the next one when that is
        ``lo``; None when no timestamp starts inside ``(lo, rows)``. Without a clock
        every row is its own timestamp.
        """
        if self.ts is None:
            return row if lo < row < self.rows else None
        first = int(np.searchsorted(self.ts, self.ts[row], "left"))
        if first > lo:
            return first
        after = int(np.searchsorted(self.ts, self.ts[row], "right"))
        return after if after < self.rows else None

    def date_starts(self, lo: int, hi: int) -> np.ndarray:
        """The first row of each date among rows ``lo`` to ``hi``, then ``hi``; without
        a clock the rows are one date.
        """
        if self.ts is None:
            return np.array([lo, hi])
        day = self.ts[lo:hi].astype("datetime64[D]")
        return np.r_[lo + np.flatnonzero(np.r_[True, day[1:] != day[:-1]]), hi]

    def middle(self, lo: int, hi: int) -> int:
        """The first row of the middle timestamp among rows ``lo`` to ``hi``."""
        stamps = np.arange(lo, hi) if self.ts is None else self.ts[lo:hi]
        first = np.unique(stamps, return_index=True)[1]
        return lo + int(first[len(first) // 2])

    def matrix(self, cols: Sequence[str], rows: Rows | None = None) -> np.ndarray:
        """Column-stacked features over a range or segments, every visible row when
        ``rows`` is None, shape (rows, len(cols)), filled one column at a time so the
        matrix is the only copy.
        """
        parts = segments((0, self.rows) if rows is None else rows)
        first = self.column(cols[0], parts)
        dtype = np.result_type(first, *(self._dtype(c) for c in cols[1:]))
        out = np.empty((len(first), len(cols)), dtype)
        out[:, 0] = first
        for j, name in enumerate(cols[1:], 1):
            out[:, j] = self.column(name, parts)
        return out

    def column(self, name: str, rows: Rows | None = None) -> np.ndarray:
        """One column over a range or segments, every visible row when ``rows`` is
        None.
        """
        parts = segments((0, self.rows) if rows is None else rows)
        for lo, hi in parts:
            if lo < 0 or hi < lo:
                raise ValueError(f"segment ({lo}, {hi}) asked; one needs 0 <= lo <= hi")
        end = max(hi for _, hi in parts)
        if end > self.rows:
            raise ValueError(
                f"rows up to {end} asked, {self.rows} visible before the cutoff"
            )
        if name in self.columns:
            values = self.columns[name]
            return np.concatenate([values[lo:hi] for lo, hi in parts])
        if name not in self.stores:
            raise KeyError(
                f"{name!r} is not a column: dataset columns {sorted(self.columns)}, "
                f"computed columns {list(self.computed_columns)}"
            )
        scan = pl.scan_parquet(self.stores[name]).select(name)
        values = np.concatenate(
            [scan.slice(lo, hi - lo).collect()[name].to_numpy() for lo, hi in parts]
        )
        if self.keep is None:
            return values
        return _blank(values, np.concatenate([self.keep[lo:hi] for lo, hi in parts]))

    def _dtype(self, name: str) -> np.dtype:
        if name in self.columns:
            return self.columns[name].dtype
        schema = pl.read_parquet_schema(self.stores[name])
        dtype = pl.Series([], dtype=schema[name]).to_numpy().dtype
        if self.keep is None:
            return dtype
        return _blank(np.empty(0, dtype), np.empty(0, bool)).dtype

    def _clock(self) -> np.ndarray:
        if self.ts is None:
            raise Refused("dataset has no timestamps; use a row-based split")
        return self.ts


@dataclasses.dataclass(frozen=True)
class FitView:
    """What ``fit`` and ``prepare`` receive: the dataset blank outside the fold's train
    segments, with the fold's context beside it.

    It reads as a ``Dataset`` through ``column``, ``matrix``, ``columns``, ``ts``,
    ``rows``, ``index_of``, ``date_at``, ``computed_columns``, ``reveal`` and
    ``auxiliary_rows``.

    Attributes
    ----------
    data : Dataset
        The dataset blank outside ``train``, from ``Dataset.train_view``.
    train : Segments
        The fold's train row segments.
    embargo_rows : int
        The evaluation split's row embargo, 0 when it has none.
    embargo_timestamps : int
        The evaluation split's timestamp embargo, 0 when it has none; with
        ``embargo_rows`` what ``splits.inner_split`` purges by.
    prepared : Any
        What the pipeline's ``prepare`` built for this train range; None without one.
    """

    data: Dataset
    train: Segments
    embargo_rows: int = 0
    embargo_timestamps: int = 0
    prepared: Any = None

    def __repr__(self) -> str:
        return f"FitView({self.data!r}, train={self.train})"

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        """The columns, blank outside ``train``."""
        return self.data.columns

    @property
    def auxiliary_rows(self) -> int:
        """The dataset's auxiliary head: that many leading rows, in every fold's
        train and scored by no evaluation, so a fit can learn from them apart from
        the rows it is judged on.
        """
        return self.data.auxiliary_rows

    @property
    def ts(self) -> np.ndarray | None:
        """The clock, or None without one."""
        return self.data.ts

    @property
    def rows(self) -> int:
        """The rows visible."""
        return self.data.rows

    @property
    def computed_columns(self) -> tuple[str, ...]:
        """The names of the computed columns."""
        return self.data.computed_columns

    @property
    def reveal(self) -> dict[str, Lag]:
        """The targets' reveal lags."""
        return self.data.reveal

    def index_of(self, when: str | datetime.date | datetime.datetime) -> int:
        """As ``Dataset.index_of``."""
        return self.data.index_of(when)

    def date_at(self, row: int) -> datetime.date:
        """As ``Dataset.date_at``."""
        return self.data.date_at(row)

    def matrix(self, cols: Sequence[str], rows: Rows | None = None) -> np.ndarray:
        """As ``Dataset.matrix``."""
        return self.data.matrix(cols, rows)

    def column(self, name: str, rows: Rows | None = None) -> np.ndarray:
        """As ``Dataset.column``."""
        return self.data.column(name, rows)


def known_before(dataset: Dataset | FitView, at: int, lag: Lag) -> int:
    """The rows whose labels are known when row ``at`` starts, as a prefix length.

    A label with a time lag ``L`` is known from ``ts + L``, one with an integer lag
    ``n`` from the first row of the n-th later date that has rows, one without a lag
    only after its own timestamp; without a clock, the rows before ``at``.
    ``Dataset.masked`` hides the rest and ``splits.inner_split`` keeps no more.

    Parameters
    ----------
    dataset : Dataset | FitView
        The rows and clock.
    at : int
        The row the labels are read at.
    lag : Lag
        The target's reveal lag: a ``np.timedelta64``, an ``int`` of dates, or None.

    Returns
    -------
    int
        The count of leading rows whose labels are known.
    """
    ts = dataset.ts
    if ts is None:
        return at
    if lag is None:
        return int(np.searchsorted(ts, ts[at], "left"))
    if isinstance(lag, np.timedelta64):
        return int(np.searchsorted(ts, ts[at] - lag, "right"))
    day = np.unique(ts.astype("datetime64[D]"), return_inverse=True)[1]
    return int(np.searchsorted(day, day[at] - lag, "right"))


def segments(rows: Rows) -> tuple[Range, ...]:
    """A range or a tuple of ranges as a tuple of ranges."""
    if isinstance(rows[0], (int, np.integer)):
        return (cast(Range, rows),)
    return cast(tuple[Range, ...], rows)


def row_mask(size: int, rows: Rows) -> np.ndarray:
    """A bool mask of ``size`` rows, True inside the segments."""
    mask = np.zeros(size, bool)
    for lo, hi in segments(rows):
        mask[lo:hi] = True
    return mask


def stratified_order(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """A row permutation that shuffles each class and interleaves the classes in
    proportion, so every contiguous block of the reordered rows holds each class in
    its share: rows sort by rank within their class over the class size, ties by
    class. Reorder the rows before ``record`` and a ``BlockedKFold`` is stratified.
    """
    classes = np.unique(y, return_inverse=True)[1]
    shuffled = np.random.default_rng(seed).permutation(len(classes))
    owner = classes[shuffled]
    by_class = np.argsort(owner, kind="stable")
    grouped = owner[by_class]
    rank = np.empty(len(owner))
    rank[by_class] = np.arange(len(owner)) - np.searchsorted(grouped, grouped)
    return shuffled[np.lexsort((owner, rank / np.bincount(owner)[owner]))]


class _Blanked(MutableMapping):
    """A train view's columns: each blank outside ``keep``, copied on its first read.

    A fit pays only for the columns it names. ``upto`` is a further view, still
    unread, so a probe's shorter prefix copies nothing either.
    """

    def __init__(self, source: Mapping[str, np.ndarray], keep: np.ndarray):
        self._source = source
        self._keep = keep
        self._read: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._read:
            self._read[name] = _blank(self._source[name], self._keep)
        return self._read[name]

    def __setitem__(self, name: str, values: np.ndarray):
        self._read[name] = values

    def __delitem__(self, name: str):
        raise TypeError("a train view's columns cannot be removed")

    def __iter__(self) -> Iterator[str]:
        added = [n for n in self._read if n not in self._source]
        return iter([*self._source, *added])

    def __len__(self) -> int:
        return len(self._source) + sum(n not in self._source for n in self._read)

    def __contains__(self, name: object) -> bool:
        return name in self._source or name in self._read

    def upto(self, row: int) -> _Blanked:
        """The first ``row`` rows, still unread."""
        if isinstance(self._source, _Blanked):
            source = self._source.upto(row)
        else:
            source = {n: a[:row] for n, a in self._source.items()}
        view = _Blanked(source, self._keep[:row])
        view._read = {n: a[:row] for n, a in self._read.items()}
        return view


def _blank(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """A copy of ``values`` blank where ``keep`` is False."""
    kind = values.dtype.kind
    text, clock = kind in "OSU", kind in "mM"
    dtype = object if text else values.dtype if clock or kind in "fc" else np.float64
    out = values.astype(dtype, copy=True)
    out[~keep] = None if text else "NaT" if clock else np.nan
    return out


# Recording ============================================================================


def record(
    ledger: Ledger,
    dataset: Dataset,
    *,
    source: str,
    targets: Sequence[str],
    params: Mapping[str, Any] | None = None,
    reveal: Mapping[str, datetime.timedelta | np.timedelta64 | int] | None = None,
    sealed_from: int | str | None = None,
    auxiliary_rows: int = 0,
    steps: Sequence[Callable] = (),
) -> str:
    """Check the targets and store the rows.

    Parameters
    ----------
    ledger : Ledger
        Where the rows and the recipe are recorded.
    dataset : Dataset
        The rows, cleaned by the caller.
    source : str
        A label for where the rows came from; ``--dataset <source>`` resolves it.
    targets : Sequence[str]
        The label columns the tool hides from fits; at least one.
    params : Mapping[str, Any], optional
        How the rows were made, recorded in the recipe as provenance.
    reveal : Mapping[str, timedelta | np.timedelta64 | int], optional
        When each target's label is known: a timedelta after its row's timestamp, or
        an int of later dates with rows (``1`` is the next trading day's open, ``2``
        two trading dates later whatever the calendar gap). A features function may
        read a revealed target, probed against that lag.
    sealed_from : int | str, optional
        The first row of the sealed tail, as a row (the first row of its timestamp)
        or a date (its first row at or after it). Only an ``Evaluation(sealed=True)``
        validates on the tail, each pipeline once, and no fold trains on it before
        such an evaluation tests it.
    auxiliary_rows : int, optional
        The auxiliary head: that many leading rows a fit may learn from and no
        evaluation scores, such as another table of the same columns ingested ahead
        of the rows a model is judged on. With a clock the head ends on the first row
        of the timestamp its count falls in, so no timestamp is split. The blocked
        splits tile the rows after it, a fold whose test range reaches into it is
        refused, and a fit reads the count as ``dataset.auxiliary_rows``.
    steps : Sequence[Callable], optional
        The module-level functions that made the rows, the cleaning included. Each
        is recorded in the recipe by its dotted path with the code keys of its module
        and the modules it imports, so the log says which code dropped the rows and
        an edit shows in the next ingest's recipe.

    Returns
    -------
    str
        The dataset id: a hash of the ``ts`` bytes, the columns, the targets, their
        lags, the sealed tail's first row and the auxiliary head's length, so an
        edit that leaves the rows as they were records nothing new.

    Raises
    ------
    Refused
        When ``targets`` is empty, or the rows are already recorded under another
        source, params or steps: the id is the rows, the recipe is the one ingest
        that wrote them.
    ValueError, KeyError, TypeError
        When the rows are empty, a lag has no clock or a bad kind, ``reveal`` names
        a non-target, a target is missing or not numeric, a column is not 1-D,
        ``sealed_from`` leaves either side empty, or ``auxiliary_rows`` is not a
        count or leaves no row to judge before the end or the sealed tail.
    """
    if not targets:
        raise Refused(
            "a dataset declares at least one target; without one nothing is masked"
        )
    if dataset.rows == 0:
        raise ValueError("a dataset needs at least one row")
    has_clock = dataset.ts is not None
    reveal = dict(reveal or {})
    if reveal and not has_clock:
        raise ValueError("a reveal lag needs a dataset with timestamps")
    if set(reveal) - set(targets):
        raise KeyError(
            f"reveal names non-targets: {sorted(set(reveal) - set(targets))}"
        )
    labels = {t: _lag(reveal[t]) if t in reveal else None for t in targets}
    window = [str(d) for d in dates.span(dataset)] if has_clock else None
    recipe = {"source": source, "params": params or {}, "targets": labels}
    if steps:
        recipe["steps"] = [_step(step) for step in steps]
    missing = [t for t in targets if t not in dataset.columns]
    if missing:
        raise KeyError(f"targets not in dataset: {missing}")
    text = [t for t in targets if not _numeric(dataset.columns[t].dtype)]
    if text:
        raise TypeError(
            f"targets {text} are not numeric; targets are masked with NaN, so encode "
            "labels as numbers"
        )
    wide = {n: a.shape for n, a in dataset.columns.items() if a.ndim != 1}
    if wide:
        raise Refused(
            f"columns of shape {wide}; a column is 1-D, one value per row, so split a "
            "2-D array into named columns"
        )
    sealed = _sealed(dataset, sealed_from)
    head = _auxiliary(dataset, auxiliary_rows, sealed)
    dataset_id = data_id(dataset, labels, sealed, head)
    canonical = identity.canonical(recipe)
    before = ledger.latest(Event.DATASET, dataset_id)
    if before is not None:
        _refuse_other_recipe(before, dataset_id, canonical)
        return dataset_id
    sha = ledger.put_blob(rows_save(dataset))
    payload = {
        "source": source,
        "window": window,
        "recipe": canonical,
        "rows": dataset.rows,
        "sealed_from": sealed,
        "auxiliary_rows": head,
        "blob": {"sha": sha, "format": formats.Format.PARQUET},
    }
    ledger.append(Event.DATASET, dataset_id, dataset_id, payload, id=dataset_id)
    return dataset_id


def _refuse_other_recipe(before: dict[str, Any], dataset_id: str, recipe: Any):
    """Refuse recording rows the log holds under another source, params or steps.

    The same recipe is the memo at work and returns the id; a different one would be
    dropped on the floor, since one event holds the rows' recipe.
    """
    stored = before["payload"]["recipe"]
    if stored == json.loads(json.dumps(recipe, default=identity.canonical)):
        return
    raise Refused(
        f"these rows are dataset {dataset_id}, recorded from source "
        f"{stored['source']!r} with another recipe; record them under that source "
        "with its params and steps, or change the rows"
    )


def _step(step: Callable) -> dict[str, Any]:
    """A cleaning function as the recipe records it.

    Its dotted path, the code keys of its module's closure hashed from the git root
    that holds it, and the distributions it imports with their versions, which is
    all a function of an installed package records.
    """
    ref = identity.function_ref(step)
    file = getattr(sys.modules[step.__module__], "__file__", None)
    if not file:
        return {"function": ref, "code_keys": {}, "distributions": {}}
    root = identity.repo_root(pathlib.Path(file))
    return {
        "function": ref,
        "code_keys": identity.code_keys((step,), root),
        "distributions": identity.imported_dists((step,), root),
    }


def _numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _sealed(dataset: Dataset, sealed_from: int | str | None) -> int | None:
    """The row the sealed tail starts at, inside the dataset, on the first row of its
    timestamp.
    """
    if sealed_from is None:
        return None
    if isinstance(sealed_from, str):
        row = dataset.index_of(sealed_from)
    else:
        row = _first_of_timestamp(dataset, int(sealed_from))
    if not 0 < row < dataset.rows:
        raise Refused(
            f"sealed_from {sealed_from!r} is row {row} of {dataset.rows}; a sealed "
            "tail starts after the first row and before the last"
        )
    return int(row)


def _auxiliary(dataset: Dataset, auxiliary_rows: int, sealed: int | None) -> int:
    """The auxiliary head's length, cut back to the first row of the timestamp its
    count falls in, with at least one row left to judge before the sealed tail or
    the end.
    """
    if isinstance(auxiliary_rows, bool) or not isinstance(
        auxiliary_rows, (int, np.integer)
    ):
        raise TypeError(
            f"auxiliary_rows is a count of leading rows, not {auxiliary_rows!r}"
        )
    head = _first_of_timestamp(dataset, int(auxiliary_rows))
    end = dataset.rows if sealed is None else sealed
    where = "the end" if sealed is None else "the sealed tail"
    if not 0 <= head < end:
        raise Refused(
            f"auxiliary_rows={auxiliary_rows} leaves no row to judge before {where} at "
            f"row {end}; the auxiliary head is followed by the rows a model is judged "
            "on"
        )
    return head


def _first_of_timestamp(dataset: Dataset, row: int) -> int:
    """``row`` moved back to the first row of its timestamp, when the dataset has a
    clock and the row is inside it.
    """
    if dataset.ts is not None and 0 < row < dataset.rows:
        return dataset.index_of(dataset.ts[row])
    return row


def _stored_lag(stored: Any) -> Lag:
    """A recipe's reveal lag back as a ``Lag``: seconds as a timedelta, ``{"dates":
    n}`` as an int, None as None.
    """
    if stored is None:
        return None
    if isinstance(stored, dict):
        return int(stored["dates"])
    return np.timedelta64(round(stored * 1e9), "ns")


def _lag(reveal: datetime.timedelta | np.timedelta64 | int) -> float | dict[str, int]:
    """Seconds for a time lag, ``{"dates": n}`` for a count of later dates."""
    if isinstance(reveal, (datetime.timedelta, np.timedelta64)):
        seconds = float(reveal / np.timedelta64(1, "s"))
        if seconds < 0:
            raise ValueError(f"a reveal lag cannot be negative: {reveal!r}")
        return seconds
    if not isinstance(reveal, (int, np.integer)):
        raise TypeError(
            f"a reveal lag is a timedelta or an int, not {type(reveal).__name__}"
        )
    if reveal < 1:
        raise Refused(
            "an integer lag counts later dates and starts at 1; a same-date reveal is "
            "a timedelta"
        )
    return {"dates": int(reveal)}


def data_id(
    dataset: Dataset,
    targets: Mapping[str, Any] | None = None,
    sealed_from: int | None = None,
    auxiliary_rows: int = 0,
) -> str:
    """The content id of a dataset: its ``ts`` bytes, column names, dtypes and values
    (a string column by its strings), which columns are targets with their reveal lags
    in seconds or a count of dates, the sealed tail's first row when it has one and
    the auxiliary head's length when it has one.
    """
    ts = None if dataset.ts is None else identity.bytes_hash(dataset.ts.tobytes())
    columns = {
        name: [str(a.dtype), identity.array_hash(a)]
        for name, a in dataset.columns.items()
    }
    sealed = {} if sealed_from is None else {"sealed_from": sealed_from}
    head = {} if not auxiliary_rows else {"auxiliary_rows": auxiliary_rows}
    return identity.content_hash(
        {"ts": ts, "columns": columns, "targets": targets, **sealed, **head}
    )


def resolve(ledger: Ledger, selector: str | None = None) -> str:
    """One recorded dataset id from a selector, the rule behind ``lab run --dataset``
    and ``Evaluation.dataset_id``.

    Parameters
    ----------
    ledger : Ledger
        The ledger holding the datasets.
    selector : str, optional
        A source, an id, or an id prefix; None for the newest dataset, allowed when
        the ledger holds one source.

    Returns
    -------
    str
        The dataset id. A source resolves to the newest dataset under it; one that
        several actors wrote under resolves to the newest the running actor recorded.

    Raises
    ------
    Refused
        When the ledger holds no dataset or several sources and no selector was
        given, when a prefix matches none or several ids, naming them, or when a
        source names datasets by several actors and none by the running actor,
        naming every actor's id, since the newest would silently be someone else's.
    """
    rows = ledger.sql("SELECT id, source, actor FROM event_dataset ORDER BY seq")
    sources = {r["source"] for r in rows}
    if selector is None:
        if len(sources) > 1:
            newest = {r["source"]: r["id"] for r in rows}
            choices = ", ".join(f"{s} ({i[:8]})" for s, i in newest.items())
            raise Refused(f"ledger holds several sources, pass --dataset: {choices}")
        if not rows:
            raise Refused(
                f"dataset (newest): 0 matches in {ledger.root.resolve()}; "
                "lab ingest first"
            )
        selector = rows[-1]["source"]
    named = [r for r in rows if r["source"] == selector]
    if len({r["actor"] for r in named}) > 1:
        own = [r for r in named if r["actor"] == actor()]
        if not own:
            raise Refused(
                f"source {selector} names datasets by several actors, none by "
                f"{actor()}; pass an id: "
                + ", ".join(f"{r['id'][:8]} ({r['actor']})" for r in named)
            )
        return own[-1]["id"]
    if named:
        return named[-1]["id"]
    matches = [r["id"] for r in rows if r["id"].startswith(selector)]
    if len(matches) != 1:
        hint = "pass a longer prefix: " + ", ".join(matches)
        raise Refused(
            f"dataset {selector}: {len(matches)} matches in {ledger.root.resolve()}; "
            f"{hint if matches else 'lab ingest first'}"
        )
    return matches[0]


def load(ledger: Ledger, dataset_id: str) -> Dataset:
    """Load a recorded dataset's rows, resident, with its targets' reveal lags.

    Parameters
    ----------
    ledger : Ledger
        The ledger the dataset was recorded in.
    dataset_id : str
        The full dataset id.

    Returns
    -------
    Dataset
        The rows, with ``reveal`` set from the recipe and ``auxiliary_rows`` from
        the record.

    Raises
    ------
    KeyError
        When no dataset has the id.
    """
    event = ledger.latest(Event.DATASET, dataset_id)
    if event is None:
        raise KeyError(f"dataset {dataset_id} not found")
    dataset = rows_load(ledger.blobs / event["payload"]["blob"]["sha"])
    targets = event["payload"]["recipe"]["targets"]
    dataset.reveal = {name: _stored_lag(lag) for name, lag in targets.items()}
    dataset.auxiliary_rows = int(event["payload"].get("auxiliary_rows") or 0)
    return dataset


# Bytes ================================================================================


def rows_save(dataset: Dataset) -> bytes:
    """A dataset as one Parquet file: ``ts`` as a nanosecond timestamp plus one column
    per array.
    """
    if TS in dataset.columns:
        raise ValueError(
            f"a dataset column cannot be named {TS!r}: ts is the clock's name"
        )
    stamps = {} if dataset.ts is None else {TS: dataset.ts}
    return formats.parquet_bytes(pl.DataFrame({**stamps, **dataset.columns}))


def rows_load(source: pathlib.Path | bytes) -> Dataset:
    """A stored dataset, read one column at a time so loading peaks near its size."""
    file = io.BytesIO(source) if isinstance(source, bytes) else source
    names = list(pl.read_parquet_schema(file))
    columns = {
        n: pl.read_parquet(file, columns=[n])[n].to_numpy() for n in names if n != TS
    }
    ts = pl.read_parquet(file, columns=[TS])[TS].to_numpy() if TS in names else None
    return Dataset(columns, ts)


def from_frame(frame: pl.DataFrame, ts: str = TS) -> Dataset:
    """A dataset from a frame; a ``ts`` timestamp column is optional."""
    stamps = frame[ts].to_numpy() if ts in frame.columns else None
    columns = {name: frame[name].to_numpy() for name in frame.columns if name != ts}
    return Dataset(columns, stamps)
