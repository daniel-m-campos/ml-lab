"""What a run refuses: leaks, moved outputs, unrevealed labels and lazy imports.

A probe reruns a function on a shorter prefix and refuses an output that moved; the
fold guards refuse a schedule that trains on what its test would hide; the import
guard refuses code the memo cannot see. Every refusal names the fix.

Examples
--------
>>> refuse_changed("gbt: predict", "prediction", full, head, at=120)  # doctest: +SKIP
Refused: gbt: predict read past row 120: ...
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import identity
from ml_lab.dataset import (
    Dataset,
    Lag,
    Range,
    row_mask,
)
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Refused
from ml_lab.splits import Fold

# Public Functions ===================================================================


def cuts(dataset: Dataset, lo: int, hi: int) -> list[int]:
    """Where a probe cuts rows ``lo`` to ``hi``: at the timestamp after the open of a
    date that opens in the range (after the range's first timestamp when none does), and
    at any timestamp inside the range, each drawn by a hash of the range. The first cut
    catches a date's opening rows reading later rows of their date, or on a daily
    calendar the next date; the second varies where in its date a cut falls from range
    to range, so no calendar lines every cut up with a date open or a fixed time of day.
    A range of one timestamp has no cut, since its function sees nothing after it.
    """
    ts = np.arange(hi) if dataset.ts is None else dataset.ts[:hi]
    stamps = lo + np.flatnonzero(np.r_[True, ts[lo + 1 : hi] != ts[lo : hi - 1]])
    if len(stamps) < 2:
        return []
    starts = dataset.date_starts(max(lo - 1, 0), hi)
    opens = starts[(starts >= lo) & (starts < stamps[-1])]
    afters = stamps[np.searchsorted(stamps, opens if opens.size else [lo], "right")]
    draw = int(identity.content_hash([lo, hi]), 16)
    inner = stamps[1:]
    return sorted(
        {int(afters[draw % len(afters)]), int(inner[(draw >> 32) % len(inner)])}
    )


def one_per_row(who: str, output: Any, rows: Range) -> np.ndarray:
    """``output`` as an array with one row per row of ``rows``, else refused."""
    values = np.asarray(output)
    n = rows[1] - rows[0]
    if values.ndim not in (1, 2) or values.shape[0] != n:
        raise Refused(
            f"{who} shape {values.shape}; it must have one row per row of the range, "
            f"shape ({n},) or ({n}, k)"
        )
    return values


def series_array(who: str, output: Any) -> np.ndarray:
    """``output`` as a series, ``(rows,)`` or ``(rows, c)`` with at least one row, else
    refused.
    """
    values = np.asarray(output)
    if values.ndim == 0:
        raise Refused(
            f"{who} shape (): the series function returned a scalar; return one row "
            "per scoring unit, a row or a day"
        )
    if values.ndim > 2 or not len(values):
        raise Refused(
            f"{who} shape {values.shape}; a series is (rows,) or (rows, c) with at "
            "least one row, one row per scoring unit, a row or a day"
        )
    return values


def noted(note: str, function: Callable, *args: Any) -> Any:
    """``function(*args)``, with ``note`` added to what it raises."""
    try:
        return function(*args)
    except Exception as error:
        error.add_note(note)
        raise


def finite_metrics(name: str, metrics: dict[str, Any], where: str) -> dict[str, Any]:
    bad = sorted(
        k
        for k, v in metrics.items()
        if not isinstance(v, (int, float, np.number)) or not np.isfinite(v)
    )
    if bad:
        raise Refused(
            f"{name}: metrics {bad} are not finite on {where}; a metric is a finite "
            "number, so return one for an empty or constant series too"
        )
    return metrics


def refuse_changed(
    who: str, name: str, full: Any, head: Any, at: int | None = None, first: int = 0
):
    """Refuse when ``head``, a function's output recomputed on the rows before ``at``
    (or, without ``at``, by the model reloaded from its saved bytes), differs from
    ``full`` by more than rounding: ``1e-9`` of the largest finite value of ``full``,
    or 1024 ulps of a narrower float dtype. Honest BLAS blocking and memory layout
    measure under 3e-12 of scale, the smallest leak 1e-7, so the band separates them
    without a per-element ulp rule, which is unsound near zero. NaN positions must
    agree. The one-sentence message names the first row that moved, both values, how
    many rows moved and the largest move as a fraction of scale, so rounding and a
    leak read differently.
    """
    dtype = np.asarray(full).dtype
    rtol = 1e-9 if dtype.kind != "f" else max(1e-9, 1024 * np.finfo(dtype).eps)
    full = np.asarray(full, dtype=np.float64)
    finite = np.abs(full[np.isfinite(full)])
    scale = float(finite.max()) if finite.size else 0.0
    full = full[: len(head)]
    head = np.asarray(head, dtype=np.float64)
    close = np.isclose(full, head, rtol=0, atol=rtol * scale, equal_nan=True)
    moved = np.flatnonzero(~close.reshape(len(head), -1).all(axis=1))
    if not moved.size:
        return
    r = moved[0]
    gap = np.abs(full[moved] - head[moved])
    worst = float(np.max(gap[np.isfinite(gap)], initial=0.0))
    row, fix = first + r, ""
    if at is None:
        where = (
            f"{who}: the model reloaded from its saved bytes predicts {name} "
            f"{_g(head[r])} at row {row} where the fitted one predicts {_g(full[r])}"
        )
        fix = ", so save must keep what predict reads and predict must be deterministic"
    else:
        where = (
            f"{who} read past row {at}: {name} at row {row} is {_g(full[r])} on the "
            f"whole input and {_g(head[r])} on the rows before {at}"
        )
    raise Refused(
        f"{where}, and {moved.size} of {len(head)} rows moved by up to {worst:.6g}, "
        f"{worst / scale if scale else np.inf:.6g} of the largest |{name}| "
        f"{scale:.6g} where rounding is allowed up to {rtol:.6g}{fix}"
    )


def refuse_overlap(folds: list[Fold]):
    for fold in folds:
        lo, hi = fold.test
        for segment in fold.train:
            if segment[0] < hi and lo < segment[1]:
                raise Refused(
                    f"fold {fold.label} trains on segment {tuple(segment)}, which "
                    f"overlaps its test range {tuple(fold.test)}"
                )


def refuse_unrevealed(dataset: Dataset, targets: dict[str, Lag], folds: list[Fold]):
    """Refuse the folds that train on a row before their test range whose label is
    not known at the range's start, naming the worst one, the lag, and the purge
    (train rows dropped before each test range) that clears every fold at once.
    """
    worst: tuple[int, int, str] = (0, 0, "")
    leaky = 0
    for fold in folds:
        train = row_mask(dataset.rows, fold.train)
        lo = fold.test[0]
        known = dataset.upto(lo + 1).masked(targets, lo)
        for name in targets:
            late = np.flatnonzero(
                train[:lo]
                & np.isnan(known.columns[name][:lo])
                & ~np.isnan(dataset.columns[name][:lo])
            )
            if not late.size:
                continue
            leaky += 1
            stamps = len(np.unique(dataset._clock()[late]))
            where = (
                f"fold {fold.label} trains on rows {late[0]} to {late[-1]} whose "
                f"{name} labels (reveal lag {_lag_text(targets[name])}) are not "
                f"known when its test range starts at row {lo}"
            )
            if (stamps, late.size) > worst[:2]:
                worst = (stamps, late.size, where)
    if leaky:
        stamps, rows, where = worst
        raise Refused(
            f"{leaky} folds train on labels revealed after their test range starts; "
            f"the worst: {where}; purge {stamps} more timestamps ({rows} rows) from "
            "the end of each train range, by raising the split's embargo_rows or "
            "embargo_timestamps, and they all clear; a label known at a later date's "
            "open takes an integer reveal lag, record(reveal={name: dates}), which is "
            "exact across holidays"
        )


def refuse_auxiliary(evaluation: Evaluation, folds: list[Fold], auxiliary_rows: int):
    """An auxiliary row is trained on and never scored: a fold whose test range
    reaches into the dataset's auxiliary head is refused.
    """
    name = evaluation.name or evaluation.id
    for fold in folds:
        lo, hi = fold.test
        if lo < auxiliary_rows:
            raise Refused(
                f"evaluation {name} fold {fold.label} validates rows {lo} to {hi}, "
                f"inside the dataset's {auxiliary_rows} auxiliary rows, which a fit "
                f"may learn from and no evaluation scores; start its folds at row "
                f"{auxiliary_rows} or later"
            )


def refuse_sealed(evaluation: Evaluation, folds: list[Fold], sealed: int | None):
    """A sealed tail is validated only by a sealed evaluation, a sealed evaluation
    needs one and validates nothing else, and a fold trains on a tail row only after
    an earlier fold of a sealed evaluation tested it.
    """
    name = evaluation.name or evaluation.id
    if evaluation.sealed and sealed is None:
        raise Refused(
            f"evaluation {name} is sealed and dataset {evaluation.dataset_id} has no "
            "sealed tail; record the dataset with sealed_from"
        )
    if sealed is None:
        return
    tested = np.zeros(max(s[1] for f in folds for s in (*f.train, f.test)), bool)
    for fold in folds:
        lo, hi = fold.test
        if evaluation.sealed and lo < sealed:
            raise Refused(
                f"evaluation {name} is sealed and fold {fold.label} validates rows "
                f"{lo} to {hi}, before row {sealed} where the dataset's sealed tail "
                f"starts; a sealed evaluation validates the tail only, so start its "
                f"folds at row {sealed} or later"
            )
        if not evaluation.sealed and hi > sealed:
            raise Refused(
                f"evaluation {name} is not sealed and fold {fold.label} validates rows "
                f"{lo} to {hi}, past row {sealed} where the dataset's sealed tail "
                f"starts; end its folds by row {sealed}, or declare "
                "Evaluation(sealed=True)"
            )
        for a, b in fold.train:
            if b > sealed and not tested[max(a, sealed) : b].all():
                raise Refused(
                    f"evaluation {name} fold {fold.label} trains on rows {a} to {b}, "
                    f"past row {sealed} where the dataset's sealed tail starts; end "
                    f"its train segments by row {sealed}, or by the end of what an "
                    "earlier fold of a sealed evaluation tested"
                )
        if evaluation.sealed:
            tested[lo:hi] = True


def frozen(prepared: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    """A prepared object with every numpy array it holds made read-only.

    Walks dicts, lists, tuples, sets, dataclasses and objects with a ``__dict__``,
    so a fit that writes into shared work raises instead of changing what the next
    fit sees. An object that holds its arrays elsewhere (slots, C state), a module
    or a class is handed over as it is, and nothing below ``FREEZE_DEPTH`` levels is
    walked; the author keeps those read-only.
    """
    seen = set() if seen is None else seen
    if id(prepared) in seen or depth > FREEZE_DEPTH:
        return prepared
    seen.add(id(prepared))
    if isinstance(prepared, np.ndarray):
        prepared.flags.writeable = False
        return prepared
    if isinstance(prepared, (types.ModuleType, type)):
        return prepared
    if isinstance(prepared, dict):
        values = list(prepared.values())
    elif isinstance(prepared, (list, tuple, set, frozenset)):
        values = list(prepared)
    elif dataclasses.is_dataclass(prepared) and not isinstance(prepared, type):
        values = [getattr(prepared, f.name) for f in dataclasses.fields(prepared)]
    elif hasattr(prepared, "__dict__") and not callable(prepared):
        values = list(vars(prepared).values())
    else:
        return prepared
    for value in values:
        frozen(value, seen, depth + 1)
    return prepared


FREEZE_DEPTH = 32


def refuse_lazy_imports(
    pipeline: Pipeline,
    root: pathlib.Path,
    dists: dict[str, str],
    loaded: set,
    stage: str,
):
    """A function that imports inside its body hides code from the memo; refuse and
    name the repo modules, the modules outside the root no distribution owns, and the
    distributions it loaded.
    """
    new = set(sys.modules) - loaded
    root = root.resolve()
    grown = sorted(
        n
        for n in new
        if (m := sys.modules.get(n)) is not None
        and (identity._module_path(m, root) or identity.unseen(m, root))
    )
    owners = identity.distribution_owners()
    tops = {name.partition(".")[0] for name in new}
    lazy = sorted({d for top in tops for d in owners.get(top, ()) if d not in dists})
    if grown or lazy:
        raise Refused(
            f"{pipeline.name or pipeline.id}: {stage} imported lazily: modules "
            f"{grown}, distributions {lazy}; import them at module level beside "
            f"{stage}, so the lock covers them, or warm a library that imports on "
            "first use there"
        )


# Private Functions ==================================================================


def _g(value: Any) -> str:
    """A float, or a row of floats, to six significant digits."""
    text = " ".join(f"{v:.6g}" for v in np.ravel(value))
    return text if np.ndim(value) == 0 else f"[{text}]"


def _lag_text(lag: Lag) -> str:
    if isinstance(lag, np.timedelta64):
        return str(datetime.timedelta(seconds=float(lag / np.timedelta64(1, "s"))))
    return f"{lag} dates"
