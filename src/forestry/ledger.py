"""The ledger: one SQLite file of JSON rows per object kind, plus a content-addressed blob store.

Rows are small and few (thousands), so filtering happens in Python over ``all(kind)``.

Examples
--------
>>> ledger = Ledger.open("/tmp/fy-example")  # doctest: +SKIP
>>> ledger.put("pipeline", "abc", {"name": "ridge"})  # doctest: +SKIP
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import time
from typing import Any, Final

from forestry import hashing


class Kinds:
    """Object kinds, one table each."""

    CAPTURE: Final = "capture"
    DATASET: Final = "dataset"
    PIPELINE: Final = "pipeline"
    EVALUATION: Final = "evaluation"
    FIT: Final = "fit"
    PREDICTIONS: Final = "predictions"
    CANDIDATE: Final = "candidate"
    SCORE: Final = "score"
    COMPARISON: Final = "comparison"
    DECISION: Final = "decision"
    BASELINE: Final = "baseline"
    DEPLOYMENT: Final = "deployment"
    CAMPAIGN: Final = "campaign"


ALL_KINDS = tuple(v for k, v in vars(Kinds).items() if k.isupper())


class Ledger:
    """Rows keyed by id per kind; decisions take an increasing integer id."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.blobs = root / "blobs" / "sha256"
        self._db = sqlite3.connect(root / "forestry.sqlite")
        self._db.execute("PRAGMA journal_mode=WAL")
        for kind in ALL_KINDS:
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {kind} (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "id TEXT UNIQUE, created REAL, body TEXT)"
            )
        self._db.commit()

    def __repr__(self) -> str:
        return f"Ledger({self.root})"

    @classmethod
    def open(cls, root: str | pathlib.Path) -> Ledger:
        """Create the directory layout if needed and open the database."""
        root = pathlib.Path(root)
        (root / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
        return cls(root)

    # Rows -----------------------------------------------------------------------------------------

    def put(self, kind: str, id: str, body: dict[str, Any]) -> bool:
        """Insert a row; returns False and writes nothing when the id exists."""
        if self.get(kind, id) is not None:
            return False
        self._db.execute(
            f"INSERT INTO {kind} (id, created, body) VALUES (?, ?, ?)",
            (id, time.time(), json.dumps({"id": id, **body}, default=_jsonable)),
        )
        self._db.commit()
        return True

    def update(self, kind: str, id: str, **changes: Any):
        """Replace fields on an existing row."""
        row = self.get(kind, id)
        if row is None:
            raise KeyError(f"{kind} {id} not found")
        row.update(changes)
        self._db.execute(
            f"UPDATE {kind} SET body = ? WHERE id = ?", (json.dumps(row, default=_jsonable), id)
        )
        self._db.commit()

    def get(self, kind: str, id: str) -> dict[str, Any] | None:
        cur = self._db.execute(f"SELECT body FROM {kind} WHERE id = ?", (str(id),))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind: str) -> list[dict[str, Any]]:
        """Every row of a kind in insertion order."""
        cur = self._db.execute(f"SELECT body FROM {kind} ORDER BY seq")
        return [json.loads(r[0]) for r in cur.fetchall()]

    def where(self, kind: str, **equals: Any) -> list[dict[str, Any]]:
        """Rows whose fields equal the given values."""
        return [r for r in self.all(kind) if all(r.get(k) == v for k, v in equals.items())]

    def next_decision_id(self) -> str:
        cur = self._db.execute(f"SELECT COUNT(*) FROM {Kinds.DECISION}")
        return str(cur.fetchone()[0] + 1)

    # Blobs ----------------------------------------------------------------------------------------

    def put_blob(self, payload: bytes) -> str:
        """Store bytes under their sha256; returns the hash."""
        sha = hashing.bytes_hash(payload)
        path = self.blobs / sha
        if not path.exists():
            path.write_bytes(payload)
        return sha

    def get_blob(self, sha: str) -> bytes:
        return (self.blobs / sha).read_bytes()


def _jsonable(obj: Any) -> Any:
    return hashing.canonical(obj)
