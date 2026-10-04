"""The ledger: one append-only SQLite table of events, SQL views over it, and a blob store.

Every write is an insert with a global ``seq``, an actor and a host. Objects carry content ids,
runs and failures carry ULIDs, so two ledgers merge by id. Blobs live under their sha256.

Examples
--------
>>> ledger = Ledger("/tmp/fy-example")  # doctest: +SKIP
>>> ledger.append("pipeline_declared", "abc", "abc", {"name": "ridge"})  # doctest: +SKIP
'abc'
"""

from __future__ import annotations

import getpass
import json
import os
import pathlib
import platform
import sqlite3
import time
from typing import Any, Final

from forestry import hashing

SCHEMA_VERSION = 9
ACTOR_ENV = "FORESTRY_ACTOR"


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


class Event:
    """Event types."""

    DATASET: Final = "dataset_recorded"
    PIPELINE: Final = "pipeline_declared"
    EVALUATION: Final = "evaluation_declared"
    RUN: Final = "run_started"
    FIT: Final = "fit_computed"
    PREDICTIONS: Final = "predictions_computed"
    ENTRY: Final = "entry_scored"
    FAILED: Final = "pipeline_failed"


DDL = """
CREATE TABLE IF NOT EXISTS event (
  seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, type TEXT NOT NULL, stream TEXT NOT NULL,
  key TEXT NOT NULL, at REAL NOT NULL, actor TEXT NOT NULL, host TEXT NOT NULL,
  payload TEXT NOT NULL CHECK (json_valid(payload)));
CREATE INDEX IF NOT EXISTS event_stream ON event(stream, type, seq);
CREATE INDEX IF NOT EXISTS event_key ON event(type, key, seq);
"""

VIEWS = """
CREATE VIEW IF NOT EXISTS dataset AS SELECT seq, id, at, actor,
  json_extract(payload,'$.process') AS process, json_extract(payload,'$.instrument') AS instrument,
  json_extract(payload,'$.window[0]') AS window_start,
  json_extract(payload,'$.window[1]') AS window_end,
  json_extract(payload,'$.rows') AS rows, json_extract(payload,'$.blob.sha') AS blob,
  json_extract(payload,'$.blob.format') AS format
FROM event WHERE type='dataset_recorded';

CREATE VIEW IF NOT EXISTS pipeline AS SELECT seq, id, at, actor,
  json_extract(payload,'$.name') AS name, json_extract(payload,'$.config') AS config,
  json_extract(payload,'$.declaration') AS declaration
FROM event WHERE type='pipeline_declared';

CREATE VIEW IF NOT EXISTS evaluation AS SELECT seq, id, at, actor,
  json_extract(payload,'$.dataset') AS dataset, json_extract(payload,'$.metrics') AS metrics,
  json_extract(payload,'$.declaration') AS declaration, json_extract(payload,'$.folds') AS folds
FROM event WHERE type='evaluation_declared';

CREATE VIEW IF NOT EXISTS run AS SELECT seq, id, at, actor, host, stream AS evaluation,
  json_extract(payload,'$.git.commit') AS "commit", json_extract(payload,'$.git.dirty') AS dirty,
  json_extract(payload,'$.git.diff.sha') AS diff, json_extract(payload,'$.env_lock') AS env_lock
FROM event WHERE type='run_started';

CREATE VIEW IF NOT EXISTS fit AS SELECT seq, id, at, actor, host, stream AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.train[0]') AS train_start,
  json_extract(payload,'$.train[1]') AS train_end,
  json_extract(payload,'$.cutoff') AS cutoff, json_extract(payload,'$.env_lock') AS env_lock,
  json_extract(payload,'$.import_shas') AS import_shas,
  json_extract(payload,'$.model.sha') AS model, json_extract(payload,'$.model.format') AS format,
  json_extract(payload,'$.duration_s') AS duration_s
FROM event WHERE type='fit_computed';

CREATE VIEW IF NOT EXISTS prediction AS SELECT seq, id, at, stream AS dataset,
  json_extract(payload,'$.fit') AS fit, json_extract(payload,'$.range[0]') AS range_start,
  json_extract(payload,'$.range[1]') AS range_end, json_extract(payload,'$.fold') AS fold,
  json_extract(payload,'$.age') AS age, json_extract(payload,'$.blob.sha') AS blob
FROM event WHERE type='predictions_computed';

CREATE VIEW IF NOT EXISTS entry AS SELECT seq, id, at, actor, stream AS evaluation,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run
FROM event WHERE type='entry_scored';

CREATE VIEW IF NOT EXISTS fold_score AS SELECT e.id AS entry, e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline, json_extract(f.value,'$.fold') AS fold,
  json_extract(f.value,'$.age') AS age, m.key AS metric, m.value AS value
FROM event e, json_each(e.payload,'$.folds') f, json_each(f.value,'$.metrics') m
WHERE e.type='entry_scored';

CREATE VIEW IF NOT EXISTS aggregate_score AS SELECT e.id AS entry, e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline, CAST(a.key AS INTEGER) AS age,
  m.key AS metric, m.value AS value
FROM event e, json_each(e.payload,'$.aggregate') a, json_each(a.value) m
WHERE e.type='entry_scored';

CREATE VIEW IF NOT EXISTS latest_entry AS SELECT evaluation, pipeline, id AS entry, run, seq
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY evaluation, pipeline ORDER BY seq DESC) AS rn
      FROM entry) WHERE rn = 1;

CREATE VIEW IF NOT EXISTS failure AS SELECT seq, id, at, actor, host, stream AS evaluation,
  key AS pipeline, json_extract(payload,'$.run') AS run, json_extract(payload,'$.error') AS error
FROM event WHERE type='pipeline_failed';
"""


class Ledger:
    """One event table plus a content-addressed blob store under a root directory."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.blobs = self.root / "blobs" / "sha256"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "forestry.sqlite")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise RuntimeError(f"ledger schema {version}, this build is {SCHEMA_VERSION}")
        self._db.executescript(DDL + VIEWS + f"PRAGMA user_version={SCHEMA_VERSION};")

    def __repr__(self) -> str:
        return f"Ledger({self.root})"

    # Events ---------------------------------------------------------------------------------------

    def append(
        self, type: str, stream: str, key: str, payload: dict[str, Any], *, id: str | None = None
    ) -> str:
        """Insert one event; an id that already exists writes nothing. Returns the id."""
        event_id = id or hashing.ulid()
        self._db.execute(
            "INSERT OR IGNORE INTO event (id, type, stream, key, at, actor, host, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                type,
                stream,
                key,
                time.time(),
                actor(),
                platform.node(),
                json.dumps(payload, default=hashing.canonical),
            ),
        )
        self._db.commit()
        return event_id

    def get(self, id: str) -> dict[str, Any] | None:
        rows = self.sql("SELECT * FROM event WHERE id = ?", (id,))
        return _event(rows[0]) if rows else None

    def events(
        self,
        type: str | None = None,
        stream: str | None = None,
        key: str | None = None,
        upto: int | None = None,
    ) -> list[dict[str, Any]]:
        """Events in seq order, filtered by any of type, stream, key and a seq bound."""
        clauses, params = [], []
        for column, value in (("type", type), ("stream", stream), ("key", key)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if upto is not None:
            clauses.append("seq <= ?")
            params.append(upto)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [_event(r) for r in self.sql(f"SELECT * FROM event {where} ORDER BY seq", params)]

    def latest(self, type: str, key: str) -> dict[str, Any] | None:
        rows = self.events(type=type, key=key)
        return rows[-1] if rows else None

    def sql(self, query: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        """Rows of any query or view as dicts."""
        return [dict(r) for r in self._db.execute(query, params).fetchall()]

    # Blobs ----------------------------------------------------------------------------------------

    def put_blob(self, payload: bytes) -> str:
        """Store bytes under their sha256 atomically; returns the sha."""
        sha = hashing.bytes_hash(payload)
        path = self.blobs / sha
        if path.exists():
            return sha
        tmp = self.blobs / f".tmp-{hashing.ulid()}"
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return sha

    def get_blob(self, sha: str) -> bytes:
        return (self.blobs / sha).read_bytes()


def actor() -> str:
    """Who is writing: ``FORESTRY_ACTOR`` or the login name."""
    return os.environ.get(ACTOR_ENV) or getpass.getuser()


def _event(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["payload"] = json.loads(out["payload"])
    return out
