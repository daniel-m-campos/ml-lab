"""The ledger: one append-only SQLite table of events, SQL views over it, and a blob
store.

Every write is an insert with a global ``seq``, an actor and a host. Objects carry
content ids, runs and failures carry ULIDs, so two ledgers merge by id. Blobs live under
their sha256. The views: ``board`` and ``head_to_head`` for reading; ``pair_fold``,
the per-fold pairs ``head_to_head`` aggregates; ``event_dataset`` to
``event_failure``, one per event type; ``score_fold``, ``score_aggregate``,
``score_latest`` and ``score_fit`` as building blocks.

Examples
--------
>>> ledger = Ledger("/tmp/ml-lab-example")  # doctest: +SKIP
>>> ledger.append("pipeline_declared", "p1", "p1", {"name": "r"})  # doctest: +SKIP
'abc'
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import sqlite3
import time
from typing import Any, Final

from ml_lab import identity
from ml_lab.identity import Refused

SCHEMA_VERSION = 18
ACTOR_ENV = "ML_LAB_ACTOR"
BUSY_TIMEOUT_S = 5.0


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
CREATE UNIQUE INDEX IF NOT EXISTS event_sealed_score
  ON event(stream, json_extract(payload,'$.pipeline'))
  WHERE type='score_recorded' AND json_extract(payload,'$.sealed');
"""

VIEWS = """
CREATE VIEW event_dataset AS SELECT seq, id, at, actor,
  json_extract(payload,'$.source') AS source,
  json_extract(payload,'$.window[0]') AS span_start,
  json_extract(payload,'$.window[1]') AS span_end,
  json_extract(payload,'$.recipe') AS recipe,
  json_extract(payload,'$.rows') AS rows, json_extract(payload,'$.blob.sha') AS blob,
  json_extract(payload,'$.blob.format') AS format
FROM event WHERE type='dataset_recorded';

CREATE VIEW event_pipeline AS SELECT seq, key AS id, at, actor,
  json_extract(payload,'$.name') AS name, json_extract(payload,'$.params') AS params,
  json_extract(payload,'$.declaration') AS declaration
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rn
      FROM event WHERE type='pipeline_declared') WHERE rn = 1;

CREATE VIEW event_evaluation AS SELECT seq, key AS id, at, actor,
  json_extract(payload,'$.name') AS name,
  json_extract(payload,'$.dataset') AS dataset,
  json_extract(payload,'$.metrics') AS directions,
  json_extract(payload,'$.declaration.scorer_params') AS scorer_params,
  json_extract(payload,'$.declaration') AS declaration,
  json_extract(payload,'$.folds') AS folds
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rn
      FROM event WHERE type='evaluation_declared') WHERE rn = 1;

CREATE VIEW event_run AS SELECT seq, id, at, actor, host, stream AS evaluation,
  json_extract(payload,'$.git.commit') AS code_commit,
  json_extract(payload,'$.git.dirty') AS dirty,
  json_extract(payload,'$.git.diff.sha') AS diff,
  json_extract(payload,'$.resolution.path') AS resolution,
  json_extract(payload,'$.resolution.sha') AS resolution_sha,
  json_extract(payload,'$.editable') AS editable,
  json_extract(payload,'$.pipelines') AS pipelines
FROM event WHERE type='run_started';

CREATE VIEW event_feature AS SELECT seq, id, at, actor, host, stream AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.columns') AS columns,
  json_extract(payload,'$.columns_id') AS columns_id,
  json_extract(payload,'$.import_shas') AS import_shas,
  json_extract(payload,'$.code_keys') AS code_keys,
  json_extract(payload,'$.probe_rows') AS probe_rows,
  json_extract(payload,'$.blob.sha') AS blob,
  json_extract(payload,'$.duration_s') AS duration_s
FROM event WHERE type='features_computed';

CREATE VIEW event_fit AS SELECT seq, id, at, actor, host, stream AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.train') AS train, json_extract(payload,'$.label') AS label,
  json_extract(payload,'$.env_lock') AS env_lock,
  json_extract(payload,'$.import_shas') AS import_shas,
  json_extract(payload,'$.code_keys') AS code_keys,
  json_extract(payload,'$.features') AS features,
  json_extract(payload,'$.model.sha') AS model,
  json_extract(payload,'$.model.format') AS format,
  json_extract(payload,'$.model.portable') AS portable,
  json_extract(payload,'$.duration_s') AS duration_s
FROM event WHERE type='fit_computed';

CREATE VIEW event_prediction AS SELECT seq, id, at, stream AS dataset,
  json_extract(payload,'$.fit') AS fit,
  json_extract(payload,'$.range[0]') AS range_start,
  json_extract(payload,'$.range[1]') AS range_end,
  json_extract(payload,'$.fold') AS fold,
  json_extract(payload,'$.blob.sha') AS blob, json_extract(payload,'$.raw') AS raw,
  json_extract(payload,'$.postprocess') AS postprocess
FROM event WHERE type='predictions_computed';

CREATE VIEW event_score AS SELECT seq, id, at, actor, stream AS evaluation,
  (SELECT json_extract(v.payload,'$.dataset') FROM event v
   WHERE v.type='evaluation_declared' AND v.key = event.stream LIMIT 1) AS dataset,
  json_extract(payload,'$.pipeline') AS pipeline, json_extract(payload,'$.run') AS run
FROM event WHERE type='score_recorded';

CREATE VIEW score_fold AS SELECT e.id AS score, e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline,
  json_extract(f.value,'$.fold') AS fold,
  json_extract(f.value,'$.label') AS label,
  m.key AS metric, m.value AS value,
  json_extract(e.payload,'$.directions."' || m.key || '"') AS direction
FROM event e, json_each(e.payload,'$.folds') f, json_each(f.value,'$.metrics') m
WHERE e.type='score_recorded';

CREATE VIEW score_aggregate AS SELECT e.id AS score,
  e.stream AS evaluation,
  json_extract(e.payload,'$.pipeline') AS pipeline,
  m.key AS metric,
  json_extract(e.payload,'$.directions."' || m.key || '"') AS direction,
  m.value AS pooled, f.n_folds, f.fold_mean,
  CASE WHEN f.n_folds > 1 THEN sqrt(f.ss / (f.n_folds - 1)) END AS fold_std,
  json_extract(e.payload,'$.series.sha') AS series,
  json_extract(e.payload,'$.series.fold_rows') AS fold_rows
FROM event e, json_each(e.payload,'$.aggregate') m
JOIN (SELECT score, metric, COUNT(value) AS n_folds, AVG(value) AS fold_mean,
             SUM((value - mean) * (value - mean)) AS ss
      FROM (SELECT *, AVG(value) OVER (PARTITION BY score, metric) AS mean
            FROM score_fold)
      GROUP BY score, metric) f
  ON f.score = e.id AND f.metric = m.key
WHERE e.type='score_recorded';

CREATE VIEW score_latest AS SELECT l.evaluation, e.name AS evaluation_name,
  l.pipeline, p.name, e.dataset, d.source, l.actor, l.score, l.run, l.seq
FROM (SELECT evaluation, pipeline, actor, id AS score, run, seq,
        ROW_NUMBER() OVER (PARTITION BY evaluation, pipeline ORDER BY seq DESC) AS rn
      FROM event_score) l
JOIN event_pipeline p ON p.id = l.pipeline
JOIN event_evaluation e ON e.id = l.evaluation
JOIN event_dataset d ON d.id = e.dataset
WHERE l.rn = 1;

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
  json_extract(f.payload,'$.run') AS run,
  json_extract(f.payload,'$.label') AS label,
  json_extract(f.payload,'$.duration_s') AS duration_s
FROM stands_on so JOIN event f ON f.type='fit_computed' AND f.id = so.fit;

CREATE VIEW board AS SELECT source, evaluation_name, name, actor, metric, direction,
  CASE WHEN direction IN ('max', 'min') AND fold_mean IS NOT NULL
    THEN RANK() OVER (PARTITION BY evaluation, metric
    ORDER BY CASE direction WHEN 'max' THEN -fold_mean ELSE fold_mean END
      NULLS LAST) END AS rank,
  fold_mean, fold_std, n_folds, pooled, evaluation, pipeline, score, run, seq
FROM (SELECT l.source, l.evaluation_name, l.name, l.actor, a.metric,
  FIRST_VALUE(a.direction) OVER (PARTITION BY l.evaluation, a.metric
    ORDER BY l.seq DESC) AS direction,
  a.fold_mean, a.fold_std, a.n_folds, a.pooled,
  l.evaluation, l.pipeline, l.score, l.run, l.seq
FROM score_latest l JOIN score_aggregate a ON a.score = l.score);

CREATE VIEW pair_fold AS WITH f AS MATERIALIZED (
  SELECT l.evaluation, l.evaluation_name, l.source, l.actor, l.pipeline, l.name,
    l.score, s.fold, s.label, s.metric, s.value,
    FIRST_VALUE(s.direction) OVER (PARTITION BY l.evaluation, s.metric
      ORDER BY l.seq DESC) AS direction
  FROM score_latest l JOIN score_fold s ON s.score = l.score)
SELECT a.evaluation, a.evaluation_name, a.source, a.actor, a.metric, a.fold, a.label,
  a.pipeline, a.name, b.pipeline AS reference, b.name AS reference_name,
  a.value, b.value AS reference_value, a.value - b.value AS delta,
  CASE a.direction WHEN 'max' THEN a.value > b.value
      WHEN 'min' THEN a.value < b.value END AS win,
  a.score, b.score AS reference_score
FROM f a JOIN f b ON b.evaluation = a.evaluation AND b.pipeline <> a.pipeline
  AND b.fold = a.fold AND b.metric = a.metric;

CREATE VIEW head_to_head AS WITH d AS (
  SELECT *, AVG(delta) OVER (PARTITION BY score, reference_score, metric) AS mean
  FROM pair_fold),
h AS (SELECT evaluation, evaluation_name, source, actor, metric, pipeline, name,
  score, reference, reference_name, reference_score,
  COUNT(delta) AS n_folds, AVG(delta) AS delta_mean,
  CASE WHEN COUNT(delta) > 1
       THEN sqrt(SUM((delta - mean) * (delta - mean)) / (COUNT(delta) - 1))
  END AS delta_std,
  SUM(win) AS wins
FROM d GROUP BY score, reference_score, metric)
SELECT h.*,
  json_extract(a.payload,'$.aggregate."' || h.metric || '"')
    - json_extract(b.payload,'$.aggregate."' || h.metric || '"') AS pooled_delta,
  delta_mean * sqrt(n_folds) / delta_std AS t
FROM h JOIN event a ON a.id = h.score JOIN event b ON b.id = h.reference_score;

CREATE VIEW event_failure AS SELECT seq, id, at, actor, host,
  stream AS evaluation,
  key AS pipeline, json_extract(payload,'$.run') AS run,
  json_extract(payload,'$.error') AS error
FROM event WHERE type='pipeline_failed' AND NOT EXISTS (
  SELECT 1 FROM event s WHERE s.type='score_recorded' AND s.stream = event.stream
    AND json_extract(s.payload,'$.pipeline') = event.key AND s.seq > event.seq);
"""


class Ledger:
    """One event table plus a content-addressed blob store under a root directory."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.blobs = self.root / "blobs" / "sha256"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            self.root / "ml_lab.sqlite", timeout=BUSY_TIMEOUT_S, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        _wal(self._db)
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise Refused(
                f"ledger schema {version}, this build is {SCHEMA_VERSION}; the log is "
                "a cache of code plus data, delete the root and rerun"
            )
        views = self._db.execute("SELECT name FROM sqlite_master WHERE type='view'")
        names = {n for (n,) in views.fetchall()} | set(re.findall(r"VIEW (\w+)", VIEWS))
        drops = "".join(f"DROP VIEW IF EXISTS {n};" for n in sorted(names))
        self._db.executescript(
            f"BEGIN IMMEDIATE; {DDL} {drops} {VIEWS} "
            f"PRAGMA user_version={SCHEMA_VERSION}; COMMIT;"
        )

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
        """Insert one event and return its id; an existing id writes nothing. A
        non-finite float is stored as JSON null.
        """
        event_id = id or identity.ulid()
        text = json.dumps(payload, default=identity.canonical)
        if "NaN" in text or "Infinity" in text:
            text = json.dumps(json.loads(text, parse_constant=lambda c: None))
        self._db.execute(
            "INSERT INTO event (id, type, stream, key, at, actor, host, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
            (
                event_id,
                type,
                stream,
                key,
                time.time(),
                actor(),
                platform.node(),
                text,
            ),
        )
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
    """Who is writing: ``ML_LAB_ACTOR``, refused when unset."""
    if not os.environ.get(ACTOR_ENV):
        raise Refused(
            f"set {ACTOR_ENV} to the person or agent writing; the ledger keeps one "
            "log for many writers"
        )
    return os.environ[ACTOR_ENV]


def _wal(db: sqlite3.Connection):
    """Switch to WAL, retrying for the busy timeout: the switch needs an exclusive
    lock, and SQLite refuses it at once, without waiting, while another connection
    holds a write lock, as when several processes open a new ledger together.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while True:
        try:
            db.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def _event(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["payload"] = json.loads(out["payload"])
    return out
