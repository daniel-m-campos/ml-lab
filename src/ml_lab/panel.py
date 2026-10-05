"""A dataset as a (time, key) grid, for cross-sectional features.

An order book is a panel over (auction second, stock), a daily universe over (date,
security). Each features function used to rebuild the grid by hand; this does it once
per dataset. Positions on the time axis are the dataset's distinct timestamps in order,
so the time axis of a prefix view's pivot is a prefix of the full one; a key first seen
after the cutoff is absent, so compare pivots through ``unpivot``, or pass ``universe=``
(every key, a constant in the function's code) so a prefix and the full dataset pivot
alike. Build a Panel inside a ``Pipeline.features`` function to grid once per dataset
rather than once per call.

Examples
--------
>>> import numpy as np
>>> from ml_lab.dataset import Dataset
>>> ts = np.array(["2025-01-01", "2025-01-01", "2025-01-02"], dtype="datetime64[s]")
>>> d = Dataset({"stock": np.array([1, 2, 1]), "px": np.array([10.0, 20.0, 11.0])}, ts)
>>> p = Panel(d, "stock")
>>> p.pivot("px")
array([[10., 20.],
       [11., nan]])
>>> p.unpivot(p.pivot("px"))
array([10., 20., 11.])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_lab.dataset import Dataset


class Panel:
    """The (time, key) axes of a dataset, computed once; pivots and their inverse."""

    def __init__(self, dataset: Dataset, key: str, universe: Any = None):
        if dataset.ts is None:
            raise ValueError("a panel needs a dataset with timestamps")
        self.times, self.ti = np.unique(dataset.ts, return_inverse=True)
        column = dataset.columns[key]
        if universe is None:
            self.universe, self.ki = np.unique(column, return_inverse=True)
        else:
            self.universe = np.unique(universe)
            self.ki = np.searchsorted(self.universe, column)
            unknown = column[~np.isin(column, self.universe)]
            if unknown.size:
                raise ValueError(f"{key} {unknown[0]!r} is not in the universe")
        width = len(self.universe)
        counts = np.bincount(self.ti * width + self.ki)
        dup = int(counts.argmax())
        if counts[dup] > 1:
            raise ValueError(
                f"rows share one (time, {key}): {self.times[dup // width]} "
                f"{self.universe[dup % width]!r}"
            )
        self.dataset = dataset

    def __repr__(self) -> str:
        return f"Panel(times={len(self.times)}, universe={len(self.universe)})"

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.times), len(self.universe)

    def pivot(
        self, name: str, fill: float = np.nan, dtype: Any = np.float64
    ) -> np.ndarray:
        """One column as a (time, key) array; a missing (time, key) holds ``fill``."""
        out = np.full(self.shape, fill, dtype=dtype)
        out[self.ti, self.ki] = self.dataset.column(name)
        return out

    def unpivot(self, pivot: np.ndarray) -> np.ndarray:
        """A (time, key) array back in dataset row order."""
        return pivot[self.ti, self.ki]
