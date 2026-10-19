"""One pipeline's feature sets: computed once per feature id, probed for a lookahead,
stored, and read on demand into the dataset its fits and predictions see.

``Features`` is the feature side of a stage. It owns the pipeline's features
functions and their ids, computes each set the log lacks behind the probe, and
assembles the dataset with the feature columns. A dry run computes and probes a set
too, so its ids are the real ones and a leak is refused before any fit, but keeps it
under the ledger's ``planned`` directory instead of the log, for the run that follows.

Examples
--------
>>> features = Features(run, "ridge", pipeline, functions)  # doctest: +SKIP
>>> features.columns_id()  # doctest: +SKIP
'3f9a...'
>>> features.visible(cut).computed_columns  # doctest: +SKIP
('lag_1', 'lag_2')
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ml_lab import formats, identity
from ml_lab.dataset import Dataset, Lag, rows_load, rows_save
from ml_lab.experiment import Pipeline
from ml_lab.guards import cuts, refuse_changed, refuse_lazy_imports
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.provenance import lock_text

if TYPE_CHECKING:
    from ml_lab.stages import RunContext

PLANNED_DAYS = 7
"""How long a dry run's feature set waits under ``planned`` for a run to use it."""


# Public Functions ===================================================================


@dataclasses.dataclass(frozen=True)
class FeatureSet:
    """One computed feature set.

    Attributes
    ----------
    columns : tuple[str, ...]
        The column names, in order.
    path : pathlib.Path | None
        The Parquet file they are read from; None in a dry run, which stores nothing.
    columns_id : str
        The id of the ordered columns' bytes.
    """

    columns: tuple[str, ...]
    path: pathlib.Path | None
    columns_id: str


class Features:
    """One pipeline's feature sets, each computed once per run and kept by id.

    Parameters
    ----------
    run : RunContext
        What every stage of the run shares: the ledger, the dataset, the targets, the
        report, the dry flag and the run's cache of sets by feature id.
    name : str
        The pipeline's name, or id, for messages.
    pipeline : Pipeline
        The pipeline whose fits read the sets; a features event records its id.
    functions : tuple[Callable, ...]
        The features functions: the pipeline's own, or the one tuple its members share.

    Attributes
    ----------
    functions : tuple[Callable, ...]
        As given.
    ids : list[str]
        One feature id per function, over the dataset, the function, its code keys and
        its environment lock.
    """

    def __init__(
        self,
        run: RunContext,
        name: str,
        pipeline: Pipeline,
        functions: tuple[Callable, ...],
    ):
        self.run = run
        self.name = name
        self.pipeline = pipeline
        self.functions = functions
        self.ids = [
            identity.content_hash(
                {
                    "dataset": run.evaluation.dataset_id,
                    "features": function,
                    "code_keys": identity.code_keys((function,), run.root),
                    "env_lock": identity.bytes_hash(
                        lock_text(
                            identity.imported_dists((function,), run.root)
                        ).encode()
                    ),
                }
            )
            for function in functions
        ]
        self.shas = identity.import_shas(functions, run.root)
        self.dists = identity.imported_dists(functions, run.root)
        self._featured: Dataset | None = None

    def columns_id(self) -> str | None:
        """The fit's feature input: the id of the ordered columns' bytes, computing the
        features first when no run has, so a dry run's ids are the real ones.
        """
        sets = self._sets()
        names = Counter(n for fs in sets for n in fs.columns)
        twice = sorted(n for n, k in names.items() if k > 1)
        if twice:
            raise Refused(
                f"{self.name}: features functions name the same columns {twice}"
            )
        return identity.content_hash([fs.columns_id for fs in sets]) if sets else None

    def visible(self, cut: int) -> Dataset:
        """The dataset's first ``cut`` rows, with the feature columns read on demand
        from their stored files.
        """
        if not self.functions:
            return self.run.dataset.upto(cut)
        if self._featured is None:
            sets = self._sets()
            names = tuple(n for fs in sets for n in fs.columns)
            stores = {
                n: fs.path for fs in sets for n in fs.columns if fs.path is not None
            }
            self._featured = Dataset(
                self.run.dataset.columns,
                self.run.dataset.ts,
                names,
                stores,
                reveal=self.run.dataset.reveal,
            )
        return self._featured.upto(cut)

    def _sets(self) -> list[FeatureSet]:
        return [self._set(f, i) for f, i in zip(self.functions, self.ids, strict=True)]

    def _set(self, function: Callable, feature_id: str) -> FeatureSet:
        """One features function's columns, computed once per feature id over the
        dataset without its targets, probed and stored. A dry run computes and probes
        them too, so the names and id are known and a leak is refused before any fit,
        but stores nothing; what it computed is kept for the process, so a dry run of
        the same features under another evaluation reuses it. The probe cuts each of
        the five spans between the sixths of the rows and the end by ``cuts``.
        """
        if feature_id in self.run.features:
            return self.run.features[feature_id]
        event = self.run.ledger.latest(Event.FEATURES, feature_id)
        if event is not None:
            payload = event["payload"]
            path = self.run.ledger.blobs / payload["blob"]["sha"]
            found = FeatureSet(tuple(payload["columns"]), path, payload["columns_id"])
            self.run.report.features_reused += 1
            return self.run.features.setdefault(feature_id, found)
        if (
            self.run.dry
            and feature_id in self.run.planned
            and feature_id in _PLANNED_FEATURES
        ):
            return self.run.features.setdefault(
                feature_id, _PLANNED_FEATURES[feature_id]
            )
        revealed: dict[str, Lag] = {
            k: v for k, v in self.run.targets.items() if v is not None
        }
        bare = Dataset(
            {
                k: v
                for k, v in self.run.dataset.columns.items()
                if k in revealed or k not in self.run.targets
            },
            self.run.dataset.ts,
        )
        ref = identity.function_ref(function)
        planned = _planned(self.run.ledger, feature_id)
        if planned is not None and self.run.dry:
            columns_id, columns, duration = planned
            self.run.log(
                f"would compute features {self.name} {ref} {len(columns)} columns: "
                f"computed and probed by an earlier dry run"
            )
            return self._keep(feature_id, columns, columns_id)
        loaded = set(sys.modules)
        if planned is None:
            self.run.log(f"computing features {self.name} {ref}")
            started = time.perf_counter()
            columns = self._columns(function, bare)
            duration = time.perf_counter() - started
        else:
            self.run.log(f"features {self.name} {ref}: from the dry run, probing")
            columns_id, columns, duration = planned
        columns_id = columns_hash(columns)
        probe_rows = self._probe(
            function, bare, revealed, columns, spans=1 if self.run.dry else 5
        )
        refuse_lazy_imports(
            self.pipeline, self.run.root, self.dists, loaded, "features"
        )
        self._unchanged()
        if self.run.dry:
            self.run.log(
                f"would compute features {self.name} {ref} {len(columns)} columns "
                f"{duration:.1f}s: {self._why(function)}; probed at rows "
                f"{probe_rows} only"
            )
            _plan(self.run.ledger, feature_id, columns, columns_id, duration)
            return self._keep(feature_id, columns, columns_id)
        run_id = self.run.start()
        sha = self.run.ledger.put_blob(rows_save(Dataset(columns, None)))
        self.run.ledger.append(
            Event.FEATURES,
            self.run.evaluation.dataset_id,
            feature_id,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "features": identity.canonical(function),
                "import_shas": identity.import_shas((function,), self.run.root),
                "code_keys": identity.code_keys((function,), self.run.root),
                "columns": list(columns),
                "columns_id": columns_id,
                "probe_rows": probe_rows,
                "duration_s": duration,
                "blob": {"sha": sha, "format": formats.Format.PARQUET},
            },
            id=feature_id,
        )
        self.run.report.features_computed += 1
        self.run.log(f"features {self.name} {len(columns)} columns {duration:.1f}s")
        for leftover in _planned_dir(self.run.ledger).glob(f"{feature_id}.*"):
            leftover.unlink(missing_ok=True)
        return self.run.features.setdefault(
            feature_id,
            FeatureSet(tuple(columns), self.run.ledger.blobs / sha, columns_id),
        )

    def _keep(
        self, feature_id: str, columns: dict[str, np.ndarray], columns_id: str
    ) -> FeatureSet:
        """A dry run's set: counted as computed, planned, and kept for the process."""
        self.run.report.features_computed += 1
        self.run.planned.add(feature_id)
        found = FeatureSet(tuple(columns), None, columns_id)
        _PLANNED_FEATURES[feature_id] = found
        return self.run.features.setdefault(feature_id, found)

    def _probe(
        self,
        function: Callable,
        bare: Dataset,
        revealed: dict[str, Lag],
        columns: dict[str, np.ndarray],
        spans: int = 5,
    ) -> list[int]:
        """Recompute the columns on the rows before each probe cut of the first
        ``spans`` of the five spans between the sixths of the rows and the end, with
        the labels not yet known at the cut hidden, and refuse a column that moves; the
        cuts. A dry run probes the first span only, the cheapest, which still catches
        a column that reads past its row or its date's later rows.
        """
        n = bare.rows
        sixths = [j * n // 6 for j in range(1, 7)]
        probe_rows = [
            at
            for span in list(zip(sixths, sixths[1:], strict=False))[:spans]
            for at in cuts(bare, *span)
        ]
        for row in probe_rows:
            head = self._columns(function, bare.upto(row).masked(revealed, row - 1))
            if set(head) != set(columns):
                raise Refused(
                    f"{self.name}: features name columns {sorted(columns)} on the "
                    f"whole dataset and {sorted(head)} on the rows before {row}"
                )
            for name, values in columns.items():
                try:
                    refuse_changed(
                        f"{self.name}: features", name, values, head[name], at=row
                    )
                except Refused as error:
                    plain = self._columns(function, bare.upto(row))[name]
                    if np.array_equal(plain, values[:row], equal_nan=True):
                        raise Refused(
                            f"{self.name}: features read a revealed target before its "
                            f"reveal lag: column {name} changes before row {row} when "
                            f"labels not yet known at row {row - 1} are hidden"
                        ) from error
                    raise
            del head
        return probe_rows

    def _why(self, function: Callable) -> str:
        """Why a feature set is computed: the inputs that moved against the nearest
        earlier computation of the same function, the one with the fewest moved.
        """
        rows = self.run.ledger.sql(
            "SELECT stream AS dataset, json_extract(payload, '$.code_keys') AS "
            "code_keys FROM event WHERE type = ? AND "
            "json_extract(payload, '$.features.__function__') = ? "
            "ORDER BY seq DESC LIMIT 64",
            (Event.FEATURES, identity.function_ref(function)),
        )
        if not rows:
            return "no earlier features of this function"
        now = identity.code_keys((function,), self.run.root)

        def moved(row: dict[str, Any]) -> list[str]:
            before = json.loads(row["code_keys"])
            files = sorted(
                k for k in set(before) | set(now) if before.get(k) != now.get(k)
            )
            return [
                reason
                for reason, differs in (
                    (
                        "dataset changed",
                        row["dataset"] != self.run.evaluation.dataset_id,
                    ),
                    (f"code changed: {', '.join(files)}", bool(files)),
                )
                if differs
            ]

        nearest = min((moved(r) for r in rows), key=len)
        return "; ".join(nearest) or "environment lock changed"

    def _columns(self, function: Callable, bare: Dataset) -> dict[str, np.ndarray]:
        out = function(bare)
        clash = sorted(set(out) & set(self.run.dataset.columns))
        if clash:
            raise Refused(f"{self.name}: features redefine dataset columns {clash}")
        columns = {}
        for name, values in out.items():
            values = np.asarray(values)
            if values.shape != (bare.rows,):
                raise Refused(
                    f"{self.name}: feature {name} has shape {values.shape}, not "
                    f"({bare.rows},)"
                )
            columns[name] = values
        return columns

    def _unchanged(self):
        """Refuse to record a sha for code that is not the code that ran."""
        now = identity.import_shas(self.functions, self.run.root)
        moved = sorted(
            k for k in set(now) | set(self.shas) if now.get(k) != self.shas.get(k)
        )
        if moved:
            raise Refused(f"{self.name}: source changed during the run: {moved}; rerun")


def columns_hash(columns: dict[str, np.ndarray]) -> str:
    """The ordered columns' names, dtypes and bytes: what a fit consumes."""
    return identity.content_hash(
        [[n, str(v.dtype), identity.array_hash(v)] for n, v in columns.items()]
    )


# Private Functions ==================================================================


_PLANNED_FEATURES: dict[str, FeatureSet] = {}
"""Feature sets a dry run computed in this process, by feature id, so a later dry run
of the same features under another evaluation reuses the names and id instead of
computing them again. The id covers the dataset, the function, its code and its
environment, so an entry is right for as long as the process runs. On disk the same
sets live under the ledger's ``planned`` directory (``_planned_dir``), where a later
process, dry or real, picks them up."""


def _planned_dir(ledger: Ledger) -> pathlib.Path:
    return ledger.root / "planned"


def _plan(
    ledger: Ledger,
    feature_id: str,
    columns: dict[str, np.ndarray],
    columns_id: str,
    duration: float,
):
    """Keep a dry run's feature set under ``planned``, outside the events and the
    blob store, so the run that follows reads it instead of computing it again;
    sets older than ``PLANNED_DAYS`` are dropped, since no run came for them.
    """
    planned = _planned_dir(ledger)
    planned.mkdir(exist_ok=True)
    stale = time.time() - PLANNED_DAYS * 86400
    for old in planned.iterdir():
        try:
            if old.stat().st_mtime < stale:
                old.unlink(missing_ok=True)
        except OSError:
            continue
    (planned / f"{feature_id}.parquet").write_bytes(rows_save(Dataset(columns, None)))
    (planned / f"{feature_id}.json").write_text(
        json.dumps(
            {
                "columns": list(columns),
                "columns_id": columns_id,
                "duration_s": duration,
            }
        )
    )


def _planned(
    ledger: Ledger, feature_id: str
) -> tuple[str, dict[str, np.ndarray], float] | None:
    """A feature set an earlier dry run left under ``planned``, when its columns
    still hash to the id it was saved with; None otherwise.
    """
    meta = _planned_dir(ledger) / f"{feature_id}.json"
    data = _planned_dir(ledger) / f"{feature_id}.parquet"
    try:
        saved = json.loads(meta.read_text())
        columns = rows_load(data).columns
    except (OSError, ValueError):
        return None
    if (
        list(columns) != saved["columns"]
        or columns_hash(columns) != saved["columns_id"]
    ):
        return None
    return saved["columns_id"], columns, float(saved["duration_s"])
