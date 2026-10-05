"""A session as a (time, key) grid, for cross-sectional features.

Both Kaggle examples are panels: Optiver is (auction second, stock), JPX is (date,
security). Each step used to rebuild the grid by hand; this does it once per session.
Positions on the time axis are the session's distinct timestamps in order, so the
time axis of a prefix view's grid is a prefix of the full one; a key first seen after
the cutoff is absent, so compare grids through ``rows``, or pass ``keys=`` (the whole
universe, a constant in the step's code) so a prefix and the full session grid alike.
Build a Panel inside a ``Pipeline.features`` step to grid once per dataset rather than
once per call.

Examples
--------
>>> import numpy as np
>>> from ml_lab.session import Session
>>> ts = np.array(["2025-01-01", "2025-01-01", "2025-01-02"], dtype="datetime64[s]")
>>> s = Session({"stock": np.array([1, 2, 1]), "px": np.array([10.0, 20.0, 11.0])}, ts)
>>> p = Panel(s, "stock")
>>> p.grid("px")
array([[10., 20.],
       [11., nan]])
>>> p.rows(p.grid("px"))
array([10., 20., 11.])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_lab.session import Session


class Panel:
    """The (time, key) axes of a session, computed once; grids and their inverse."""

    def __init__(self, session: Session, key: str, keys: Any = None):
        if session.ts is None:
            raise ValueError("a panel needs a session with timestamps")
        self.times, self.ti = np.unique(session.ts, return_inverse=True)
        column = session.columns[key]
        if keys is None:
            self.keys, self.ki = np.unique(column, return_inverse=True)
        else:
            self.keys = np.unique(keys)
            self.ki = np.searchsorted(self.keys, column)
            unknown = column[~np.isin(column, self.keys)]
            if unknown.size:
                raise ValueError(f"{key} {unknown[0]!r} is not in keys")
        self.session = session

    def __repr__(self) -> str:
        return f"Panel(times={len(self.times)}, keys={len(self.keys)})"

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.times), len(self.keys)

    def grid(
        self, name: str, fill: float = np.nan, dtype: Any = np.float64
    ) -> np.ndarray:
        """One column as a (time, key) array; a missing (time, key) holds ``fill``."""
        out = np.full(self.shape, fill, dtype=dtype)
        out[self.ti, self.ki] = self.session.columns[name]
        return out

    def rows(self, grid: np.ndarray) -> np.ndarray:
        """A (time, key) array back in session row order."""
        return grid[self.ti, self.ki]
