"""Uncertainty over the stored series: two scores of one evaluation paired row by row.

The views pair scores fold by fold; at one fold, or to read the noise of a pooled
metric, pair the series a score stores. A metric that adds over rows (a PnL, a summed
loss) has the sum of row differences as its pooled difference.

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
    """

    n: int
    pooled_delta: float
    se_iid: float
    se_block: float | None


def paired(
    ledger: Ledger,
    score_a: str,
    score_b: str,
    column: str,
    block_rows: int | None = None,
) -> Paired:
    """Pair one column of two scores' series row by row.

    The block bootstrap draws the blocks with replacement, ``RESAMPLES`` times under a
    fixed seed, so one call returns one number; the last block may be short. Scores
    whose series cover other rows are refused.
    """
    (a, rows_a), (b, rows_b) = (_series(ledger, s, column) for s in (score_a, score_b))
    if rows_a != rows_b:
        raise Refused(
            "the two scores cover different rows; pair scores of one evaluation"
        )
    d = a - b
    se_block = None
    if block_rows is not None:
        sums = np.add.reduceat(d, np.arange(0, len(d), block_rows))
        draws = np.random.default_rng(SEED).integers(
            0, len(sums), (RESAMPLES, len(sums))
        )
        se_block = float(sums[draws].sum(axis=1).std(ddof=1))
    return Paired(
        len(d), float(d.sum()), float(np.sqrt(len(d)) * d.std(ddof=1)), se_block
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
