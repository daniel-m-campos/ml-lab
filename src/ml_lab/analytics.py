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
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ml_lab import formats
from ml_lab.ledger import Event, Ledger, Refused

RESAMPLES = 1000
SEED = 0


@dataclasses.dataclass(frozen=True)
class Paired:
    """The row differences of two series, summed, with their standard errors.

    Attributes
    ----------
    n : int
        Rows paired.
    pooled_delta : float
        The sum of the row differences, the first score's minus the second's.
    se_iid : float
        ``sqrt(n) * d.std(ddof=1)``, its standard error with independent rows.
    se_block : float | None
        Its standard error by a paired block bootstrap over contiguous blocks of
        ``block_rows`` rows, None without ``block_rows``.
    fold_deltas : tuple[float, ...]
        The sum of the row differences within each of the first score's folds, so a
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

    The block bootstrap draws the blocks with replacement, ``RESAMPLES`` times under a
    fixed seed, so one call returns one number; the last block may be short. Two
    scores whose evaluations store the same folds (one evaluation, or a sweep of
    its scorer params) pair as stored, overlapping test windows included; two whose
    folds differ pair when their test ranges cover the same rows in row order, and
    are refused otherwise, as are scores on different datasets and series that differ
    in length.
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


def _evaluation(ledger: Ledger, score: str) -> str:
    return ledger.latest(Event.SCORE, score)["stream"]


def _dataset(ledger: Ledger, score: str) -> str:
    evaluation = ledger.latest(Event.EVALUATION, _evaluation(ledger, score))
    return evaluation["payload"]["dataset"]


def _tests(ledger: Ledger, score: str) -> list[list[int]]:
    """The test ranges of a score's evaluation, in fold order."""
    folds = ledger.latest(Event.EVALUATION, _evaluation(ledger, score))["payload"][
        "folds"
    ]
    return [f["test"] for f in folds]


def _covered(ledger: Ledger, score: str) -> list[list[int]]:
    """The rows a score's folds test, as merged ranges; refused when the folds are
    not in row order, since the series then does not read as one row sequence.
    """
    folds = ledger.latest(Event.EVALUATION, _evaluation(ledger, score))["payload"][
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
