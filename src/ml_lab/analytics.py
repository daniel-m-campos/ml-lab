"""Uncertainty over the stored series: two scores paired row by row.

The views pair scores fold by fold; at one fold, or to read the noise of a pooled
metric, pair the series a score stores. A metric that adds over rows (a PnL, a summed
loss) has the sum of row differences as its pooled difference. Two scores pair when
their folds' test ranges cover the same rows in row order, so a refit-cadence twin of
an evaluation, the same rows under finer folds, pairs against the house score.

Examples
--------
>>> result = paired(ledger, score_a, score_b, "pnl", block_rows=500)  # doctest: +SKIP
>>> result.pooled_delta / result.se_block  # doctest: +SKIP
2.1
>>> paired_sets(ledger, evaluation, "sharpe", seeds_b, seeds_a).t  # doctest: +SKIP
1.7
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_lab import formats
from ml_lab.ledger import Event, Ledger, Refused

RESAMPLES = 1000
SEED = 0


@dataclasses.dataclass(frozen=True)
class Paired:
    """Two series paired row by row: the summed differences and their standard errors.

    Attributes
    ----------
    n : int
        Rows paired.
    pooled_delta : float
        The sum of the row differences, the first score's minus the second's.
    se_iid : float
        ``sqrt(n) * d.std(ddof=1)``: the standard error with independent rows.
    se_block : float | None
        The standard error by a paired block bootstrap over contiguous blocks of
        ``block_rows`` rows; None without ``block_rows``.
    fold_deltas : tuple[float, ...]
        The row differences summed within each of the first score's folds, so a
        finer-grained twin regroups into the house folds when the house score is
        passed first.
    """

    n: int
    pooled_delta: float
    se_iid: float
    se_block: float | None
    fold_deltas: tuple[float, ...]


def paired(
    ledger: Ledger,
    score_a: str,
    score_b: str,
    column: str,
    block_rows: int | None = None,
) -> Paired:
    """Pair one column of two scores' series row by row.

    Two scores of one evaluation, or of two evaluations with the same folds, pair as
    stored, overlapping test windows included. Two scores of different evaluations
    pair when their folds' test ranges cover the same rows in row order. The block
    bootstrap draws blocks with replacement ``RESAMPLES`` times under a fixed seed,
    so one call returns one number; the last block may be short.

    Parameters
    ----------
    ledger : Ledger
        The ledger holding both scores.
    score_a, score_b : str
        Score ids; the differences are ``score_a`` minus ``score_b``.
    column : str
        The series column to pair, as the scorer named it.
    block_rows : int, optional
        Rows per bootstrap block; None skips the block bootstrap.

    Returns
    -------
    Paired
        The paired differences with their standard errors.

    Raises
    ------
    Refused
        When a score or column is missing, the scores are on different datasets,
        their folds cover different rows, or the series differ in length.
    """
    (a, rows_a), (b, rows_b) = (_series(ledger, s, column) for s in (score_a, score_b))
    if _dataset(ledger, score_a) != _dataset(ledger, score_b):
        raise Refused("the two scores are on different datasets; pair rows of one")
    same = _tests(ledger, score_a) == _tests(ledger, score_b)
    if not same and _covered(ledger, score_a) != _covered(ledger, score_b):
        raise Refused(
            "the two scores cover different rows; pair scores of one evaluation, or "
            "of two whose folds' test ranges concatenate to the same rows"
        )
    if len(a) != len(b):
        raise Refused(
            f"the two scores store series of {len(a)} and {len(b)} rows over the "
            "same rows; a series pairs row by row"
        )
    d = a - b
    se_block = None
    if block_rows is not None:
        sums = np.add.reduceat(d, np.arange(0, len(d), block_rows))
        draws = np.random.default_rng(SEED).integers(
            0, len(sums), (RESAMPLES, len(sums))
        )
        se_block = float(sums[draws].sum(axis=1).std(ddof=1))
    cuts = np.cumsum([0, *rows_a[:-1]])
    folds = tuple(float(v) for v in np.add.reduceat(d, cuts))
    return Paired(
        len(d), float(d.sum()), float(np.sqrt(len(d)) * d.std(ddof=1)), se_block, folds
    )


@dataclasses.dataclass(frozen=True)
class SetPaired:
    """Two sets of pipelines paired fold by fold on their set means.

    Attributes
    ----------
    n_folds : int
        Folds where every member of both sets has the metric.
    delta_mean : float
        The mean over folds of the first set's mean metric minus the second's.
    delta_std : float | None
        Its sample standard deviation over folds; None at one fold.
    t : float | None
        ``delta_mean * sqrt(n_folds) / delta_std``; None at one fold or zero spread.
    wins : int | None
        Folds the first set wins by the metric's direction; None without one.
    fold_deltas : tuple[float, ...]
        The per-fold differences of the set means, in fold order.
    """

    n_folds: int
    delta_mean: float
    delta_std: float | None
    t: float | None
    wins: int | None
    fold_deltas: tuple[float, ...]


def paired_sets(
    ledger: Ledger,
    evaluation: str,
    metric: str,
    pipelines: Sequence[str],
    references: Sequence[str],
) -> SetPaired:
    """Pair a set of pipelines against a set of references under one evaluation.

    A seed set against a seed set, or a post-process over seeds against the seeds.
    Each fold's value for a set is the mean of its members' latest scores on
    ``metric``, so the fold-paired t respects the folds the members share instead of
    treating each member as another sample.

    Parameters
    ----------
    ledger : Ledger
        The ledger holding the scores.
    evaluation : str
        The evaluation id both sets were scored under.
    metric : str
        The metric name, as the scorer's ``metrics`` returns it.
    pipelines, references : Sequence[str]
        Pipeline ids of the two sets; the differences are ``pipelines`` minus
        ``references``.

    Returns
    -------
    SetPaired
        The fold-paired comparison of the two sets.

    Raises
    ------
    Refused
        When a set is empty, a member has no latest score under the evaluation, or
        the members' scores do not share one set of folds.
    """
    if not pipelines or not references:
        raise Refused("paired_sets needs at least one pipeline id in each set")
    values = _fold_values(ledger, evaluation, metric, [*pipelines, *references])
    folds = sorted(next(iter(values.values())))
    a = np.array([np.mean([values[p][f] for p in pipelines]) for f in folds])
    b = np.array([np.mean([values[p][f] for p in references]) for f in folds])
    d = a - b
    direction = ledger.sql(
        "SELECT direction FROM score_fold s JOIN score_latest l ON l.score = s.score "
        "WHERE l.evaluation = ? AND s.metric = ? ORDER BY l.seq DESC LIMIT 1",
        (evaluation, metric),
    )[0]["direction"]
    wins = None
    if direction in ("max", "min"):
        wins = int(((d > 0) if direction == "max" else (d < 0)).sum())
    std = float(d.std(ddof=1)) if len(d) > 1 else None
    t = float(d.mean() * np.sqrt(len(d)) / std) if std else None
    return SetPaired(len(d), float(d.mean()), std, t, wins, tuple(map(float, d)))


# Private Functions ====================================================================


def _series(ledger: Ledger, score: str, column: str) -> tuple[np.ndarray, list[int]]:
    event = ledger.latest(Event.SCORE, score)
    if event is None:
        raise Refused(f"score {score} is not in {ledger.root}; pass a score id")
    stored = event["payload"]["series"]
    arrays = formats.arrays_load(ledger.get_blob(stored["sha"]))
    if column not in arrays:
        raise Refused(f"score {score} stores series {list(arrays)}, not {column}")
    return arrays[column], stored["fold_rows"]


def _fold_values(
    ledger: Ledger, evaluation: str, metric: str, pipelines: Sequence[str]
) -> dict[str, dict[int, float]]:
    """Per pipeline id, its latest score's metric per fold under the evaluation.

    Refused when a pipeline has no such score or the folds differ.
    """
    marks = ", ".join("?" * len(pipelines))
    rows = ledger.sql(
        "SELECT l.pipeline, s.fold, s.value FROM score_latest l "
        "JOIN score_fold s ON s.score = l.score "
        f"WHERE l.evaluation = ? AND s.metric = ? AND l.pipeline IN ({marks})",
        (evaluation, metric, *pipelines),
    )
    values: dict[str, dict[int, float]] = {p: {} for p in pipelines}
    for row in rows:
        if row["value"] is not None:
            values[row["pipeline"]][row["fold"]] = row["value"]
    missing = [p for p in pipelines if not values[p]]
    if missing:
        raise Refused(
            f"no latest score of {metric} under evaluation {evaluation} for "
            f"pipelines {missing}; pass pipeline ids the evaluation scored"
        )
    folds = {p: set(v) for p, v in values.items()}
    if len({frozenset(f) for f in folds.values()}) > 1:
        raise Refused(
            f"the pipelines' scores of {metric} do not share one set of folds: "
            + ", ".join(f"{p[:8]} has {sorted(f)}" for p, f in folds.items())
        )
    return values


def _latest(ledger: Ledger, kind: str, key: str) -> dict[str, Any]:
    """The newest event of a kind under a key, refused when there is none."""
    event = ledger.latest(kind, key)
    if event is None:
        raise Refused(f"{kind} {key} is not in {ledger.root}")
    return event


def _evaluation(ledger: Ledger, score: str) -> str:
    """The evaluation id a score was recorded under."""
    return _latest(ledger, Event.SCORE, score)["stream"]


def _dataset(ledger: Ledger, score: str) -> str:
    """The dataset id of a score's evaluation."""
    evaluation = _latest(ledger, Event.EVALUATION, _evaluation(ledger, score))
    return evaluation["payload"]["dataset"]


def _tests(ledger: Ledger, score: str) -> list[list[int]]:
    """The test ranges of a score's evaluation, in fold order."""
    folds = _latest(ledger, Event.EVALUATION, _evaluation(ledger, score))["payload"][
        "folds"
    ]
    return [f["test"] for f in folds]


def _covered(ledger: Ledger, score: str) -> list[list[int]]:
    """The rows a score's folds test, as merged ranges.

    Refused when the folds are not in row order, since the series then does not
    read as one row sequence.
    """
    folds = _latest(ledger, Event.EVALUATION, _evaluation(ledger, score))["payload"][
        "folds"
    ]
    merged: list[list[int]] = []
    for lo, hi in (f["test"] for f in folds):
        if merged and lo < merged[-1][1]:
            raise Refused(
                f"score {score}: its evaluation's folds are not in row order; pair "
                "scores of one evaluation"
            )
        if merged and lo == merged[-1][1]:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return merged
