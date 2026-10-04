"""The ledger: one append-only SQLite table of events, SQL views over it, and a blob
store.

Every write is an insert with a global ``seq``, an actor and a host. Objects carry
content ids, runs and failures carry ULIDs, so two ledgers merge by id. Blobs live under
their sha256.

Examples
--------
>>> ledger = Ledger("/tmp/fy-example")  # doctest: +SKIP
>>> ledger.append("pipeline_declared", "p1", "p1", {"name": "r"})  # doctest: +SKIP
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

from ml_lab import identity

SCHEMA_VERSION = 14
ACTOR_ENV = "ML_LAB_ACTOR"


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


class Event:
    """Event types."""

    DATASET: Final = "dataset_recorded"
    PIPELINE: Final = "pipeline_declared"
    EVALUATION: Final = "evaluation_declared"
    RUN: Final = "run_started"
    FEATURES: Final = "features_computed"
    FIT: Final = "fit_computed"
    PREDICTIONS: Final = "predictions_computed"
    SCORE: Final = "score_recorded"
    FAILED: Final = "pipeline_failed"


DDL = """
CREATE TABLE IF NOT EXISTS event (
  seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, type TEXT NOT NULL,
  stream TEXT NOT NULL,
  key TEXT NOT NULL, at REAL NOT NULL, actor TEXT NOT NULL, host TEXT NOT NULL,
  payload TEXT NOT NULL CHECK (json_valid(payload)));
CREATE INDEX IF NOT EXISTS event_stream ON event(stream, type, seq);
CREATE INDEX IF NOT EXISTS event_key ON event(type, key, seq);
"""

VIEWS = """
DROP VIEW IF EXISTS dataset;
CREATE VIEW dataset AS SELECT seq, id, at, actor,
  json_extract(payload,'$.source') AS source,
  json_extract(payload,'$.window[0]') AS window_start,
  json_extract(payload,'$.window[1]') AS window_end,
  json_extract(payload,'$.rows') AS rows, json_extract(payload,'$.blob.sha') AS blob,
  json_extract(payload,'$.blob.format') AS format
FROM event WHERE type='dataset_recorded';

DROP VIEW IF EXISTS pipeline;
CREATE VIEW pipeline AS SELECT seq, key AS id, at, actor,
  json_extract(payload,'$.name') AS name, json_extract(payload,'$.config') AS config,
  json_extract(payload,'$.declaration') AS declaration
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rn
      FROM event WHERE type='pipeline_declared') WHERE rn = 1;

DROP VIEW IF EXISTS evaluation;
CREATE VIEW evaluation AS SELECT seq, key AS id, at, actor,
  json_extract(payload,'$.name') AS name,
  json_extract(payload,'$.dataset') AS dataset,
  json_extract(payload,'$.metrics') AS metrics,
  json_extract(payload,'$.declaration') AS declaration,
  json_extract(payload,'$.folds') AS folds
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rn
      FROM event WHERE type='evaluation_declared') WHERE rn = 1;

DROP VIEW IF EXISTS run;
CREATE VIEW run AS SELECT seq, id, at, actor, host, stream AS evaluation,
  json_extract(payload,'$.git.commit') AS "commit",
  json_extract(payload,'$.git.dirty') AS dirty,
  json_extract(payload,'$.git.diff.sha') AS diff,
  json_extract(payload,'$.resolution.path') AS resolution,
  json_extract(payload,'$.resolution.sha') AS resolution_sha,
  json_extract(payload,'$.pipelines') AS pipelines
FROM event WHERE type='run_started';

DROP VIEW IF EXISTS feature;
CREATE VIEW feature AS SELECT seq, id, at, actor, host, stream AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.columns') AS columns,
  json_extract(payload,'$.import_shas') AS import_shas,
  json_extract(payload,'$.probe_at') AS probe_at,
  json_extract(payload,'$.blob.sha') AS blob,
  json_extract(payload,'$.duration_s') AS duration_s
FROM event WHERE type='features_computed';

DROP VIEW IF EXISTS fit;
CREATE VIEW fit AS SELECT seq, id, at, actor, host, stream AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.train') AS train, json_extract(payload,'$.label') AS label,
  json_extract(payload,'$.env_lock') AS env_lock,
  json_extract(payload,'$.import_shas') AS import_shas,
  json_extract(payload,'$.model.sha') AS model,
  json_extract(payload,'$.model.format') AS format,
  json_extract(payload,'$.model.portable') AS portable,
  json_extract(payload,'$.duration_s') AS duration_s
FROM event WHERE type='fit_computed';

DROP VIEW IF EXISTS prediction;
CREATE VIEW prediction AS SELECT seq, id, at, stream AS dataset,
  json_extract(payload,'$.fit') AS fit,
  json_extract(payload,'$.range[0]') AS range_start,
  json_extract(payload,'$.range[1]') AS range_end,
  json_extract(payload,'$.fold') AS fold,
  json_extract(payload,'$.window') AS window,
  json_extract(payload,'$.blob.sha') AS blob, json_extract(payload,'$.raw') AS raw,
  json_extract(payload,'$.postprocess') AS postprocess
FROM event WHERE type='predictions_computed';

DROP VIEW IF EXISTS score;
CREATE VIEW score AS SELECT seq, id, at, actor, stream AS evaluation,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run
FROM event WHERE type='score_recorded';

DROP VIEW IF EXISTS fold_score;
CREATE VIEW fold_score AS SELECT e.id AS score, e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline,
  json_extract(f.value,'$.fold') AS fold,
  json_extract(f.value,'$.label') AS label, json_extract(f.value,'$.window') AS window,
  m.key AS metric, m.value AS value
FROM event e, json_each(e.payload,'$.folds') f, json_each(f.value,'$.metrics') m
WHERE e.type='score_recorded';

DROP VIEW IF EXISTS aggregate_score;
CREATE VIEW aggregate_score AS SELECT e.id AS score,
  e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline, a.key AS window,
  m.key AS metric, m.value AS value, f.folds, f.fold_mean,
  CASE WHEN f.folds > 1
       THEN sqrt(max(f.sq - f.fold_mean * f.fold_mean, 0) * f.folds / (f.folds - 1))
  END AS fold_std,
  json_extract(e.payload,'$.series.' || a.key || '.sha') AS series
FROM event e, json_each(e.payload,'$.aggregate') a, json_each(a.value) m
JOIN (SELECT score, window, metric, COUNT(*) AS folds, AVG(value) AS fold_mean,
             AVG(value * value) AS sq
      FROM fold_score GROUP BY score, window, metric) f
  ON f.score = e.id AND f.window = a.key AND f.metric = m.key
WHERE e.type='score_recorded';

DROP VIEW IF EXISTS latest_score;
CREATE VIEW latest_score AS SELECT l.evaluation, e.name AS evaluation_name,
  l.pipeline, p.name, e.dataset, d.source, l.score, l.run, l.seq
FROM (SELECT evaluation, pipeline, id AS score, run, seq,
        ROW_NUMBER() OVER (PARTITION BY evaluation, pipeline ORDER BY seq DESC) AS rn
      FROM score) l
JOIN pipeline p ON p.id = l.pipeline
JOIN evaluation e ON e.id = l.evaluation
JOIN dataset d ON d.id = e.dataset
WHERE l.rn = 1;

DROP VIEW IF EXISTS score_fit;
CREATE VIEW score_fit AS WITH RECURSIVE stands_on(score, evaluation, pipeline, fit) AS (
  SELECT s.id, s.stream, json_extract(s.payload,'$.pipeline'),
         json_extract(pr.payload,'$.fit')
  FROM event s, json_each(s.payload,'$.predictions') p
  JOIN event pr ON pr.type='predictions_computed' AND pr.id = p.value
  WHERE s.type='score_recorded'
  UNION
  SELECT so.score, so.evaluation, so.pipeline, m.value
  FROM stands_on so
  JOIN event f ON f.type='fit_computed' AND f.id = so.fit,
  json_each(f.payload,'$.members') m
)
SELECT DISTINCT so.score, so.evaluation, so.pipeline, f.id AS fit,
  json_extract(f.payload,'$.pipeline') AS fit_pipeline,
  json_extract(f.payload,'$.label') AS label,
  json_extract(f.payload,'$.duration_s') AS duration_s
FROM stands_on so JOIN event f ON f.type='fit_computed' AND f.id = so.fit;

DROP VIEW IF EXISTS paired_score;
CREATE VIEW paired_score AS SELECT la.evaluation, a.window, a.metric,
  la.pipeline, la.name, la.score,
  lb.pipeline AS reference, lb.name AS reference_name, lb.score AS reference_score,
  COUNT(*) AS folds, AVG(a.value - b.value) AS mean_delta,
  CASE WHEN COUNT(*) > 1
       THEN sqrt(max(AVG((a.value - b.value) * (a.value - b.value))
                     - AVG(a.value - b.value) * AVG(a.value - b.value), 0)
                 * COUNT(*) / (COUNT(*) - 1))
  END AS delta_std,
  SUM(CASE json_extract(v.metrics, '$.' || a.metric)
      WHEN 'max' THEN a.value > b.value WHEN 'min' THEN a.value < b.value END) AS wins
FROM latest_score la
JOIN latest_score lb ON lb.evaluation = la.evaluation AND lb.pipeline <> la.pipeline
JOIN fold_score a ON a.score = la.score
JOIN fold_score b ON b.score = lb.score AND b.fold = a.fold
  AND b.window = a.window AND b.metric = a.metric
JOIN evaluation v ON v.id = la.evaluation
GROUP BY la.score, lb.score, a.window, a.metric;

DROP VIEW IF EXISTS failure;
CREATE VIEW failure AS SELECT seq, id, at, actor, host,
  stream AS evaluation,
  key AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.error') AS error
FROM event WHERE type='pipeline_failed';
"""


class Ledger:
    """One event table plus a content-addressed blob store under a root directory."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.blobs = self.root / "blobs" / "sha256"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "ml_lab.sqlite")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise Refused(
                f"ledger schema {version}, this build is {SCHEMA_VERSION}; the log is "
                "a cache of code plus data, delete the root and rerun"
            )
        self._db.executescript(DDL + VIEWS + f"PRAGMA user_version={SCHEMA_VERSION};")

    def __repr__(self) -> str:
        return f"Ledger({self.root})"

    # Events ---------------------------------------------------------------------------

    def append(
        self,
        type: str,
        stream: str,
        key: str,
        payload: dict[str, Any],
        *,
        id: str | None = None,
    ) -> str:
        """Insert one event and return its id; an existing id writes nothing."""
        event_id = id or identity.ulid()
        self._db.execute(
            "INSERT OR IGNORE INTO event "
            "(id, type, stream, key, at, actor, host, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                type,
                stream,
                key,
                time.time(),
                actor(),
                platform.node(),
                json.dumps(payload, default=identity.canonical),
            ),
        )
        self._db.commit()
        return event_id

    def events(
        self,
        type: str | None = None,
        stream: str | None = None,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Events in seq order, filtered by any of type, stream and key."""
        clauses, params = [], []
        for column, value in (("type", type), ("stream", stream), ("key", key)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [
            _event(r)
            for r in self.sql(f"SELECT * FROM event {where} ORDER BY seq", params)
        ]

    def latest(self, type: str, key: str) -> dict[str, Any] | None:
        rows = self.events(type=type, key=key)
        return rows[-1] if rows else None

    def sql(self, query: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        """Rows of any query or view as dicts."""
        return [dict(r) for r in self._db.execute(query, params).fetchall()]

    # Blobs ----------------------------------------------------------------------------

    def put_blob(self, payload: bytes) -> str:
        """Store bytes under their sha256 atomically; returns the sha."""
        sha = identity.bytes_hash(payload)
        path = self.blobs / sha
        if path.exists():
            return sha
        tmp = self.blobs / f".tmp-{identity.ulid()}"
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return sha

    def get_blob(self, sha: str) -> bytes:
        return (self.blobs / sha).read_bytes()


def actor() -> str:
    """Who is writing: ``ML_LAB_ACTOR`` or the login name."""
    return os.environ.get(ACTOR_ENV) or getpass.getuser()


def _event(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["payload"] = json.loads(out["payload"])
    return out
