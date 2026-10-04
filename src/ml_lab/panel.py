"""A session as a (time, key) grid, for cross-sectional features.

Both Kaggle examples are panels: Optiver is (auction second, stock), JPX is (date,
security). Each step used to rebuild the grid by hand; this does it once per session.
Positions on the time axis are the session's distinct timestamps in order, so a grid
of a prefix view is a prefix of the full grid and the lookahead guard still holds.

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

import numpy as np

from ml_lab.session import Session


class Panel:
    """The (time, key) axes of a session, computed once; grids and their inverse."""

    def __init__(self, session: Session, key: str):
        if session.ts is None:
            raise ValueError("a panel needs a session with timestamps")
        self.times, self.ti = np.unique(session.ts, return_inverse=True)
        self.keys, self.ki = np.unique(session.columns[key], return_inverse=True)
        self.session = session

    def __repr__(self) -> str:
        return f"Panel(times={len(self.times)}, keys={len(self.keys)})"

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.times), len(self.keys)

    def grid(self, name: str, fill: float = np.nan) -> np.ndarray:
        """One column as a (time, key) array; a missing (time, key) holds ``fill``."""
        out = np.full(self.shape, fill, dtype=np.float64)
        out[self.ti, self.ki] = self.session.columns[name]
        return out

    def rows(self, grid: np.ndarray) -> np.ndarray:
        """A (time, key) array back in session row order."""
        return grid[self.ti, self.ki]
