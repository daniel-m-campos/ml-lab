"""One pipeline through a run: its memoized feature sets, fits and predictions.

``RunContext`` is what every stage of one run shares; ``Stage`` computes one
pipeline's ids, looks each up, and computes and records what the log lacks, behind
the probes in ``ml_lab.guards``. A blend holds one stage per member.

Examples
--------
>>> stage = context.stage(pipeline)  # doctest: +SKIP
>>> fit_id = stage.fit(fold, 0)  # doctest: +SKIP
>>> stage.predictions(fit_id, fold, 0, fold.test)  # doctest: +SKIP
'7c1e...'
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import sys
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import formats, identity
from ml_lab.dataset import (
    Dataset,
    FitView,
    Lag,
    Range,
    Segments,
    rows_load,
    rows_save,
)
from ml_lab.declarations import positional, stage_functions
from ml_lab.experiment import Evaluation, Fitted, Pipeline
from ml_lab.guards import (
    cuts,
    frozen,
    noted,
    one_per_row,
    refuse_changed,
    refuse_lazy_imports,
)
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.provenance import lock_text
from ml_lab.splits import Fold

# Public Functions ===================================================================


@dataclasses.dataclass
class RunReport:
    """What a run wrote; ``run`` is empty when every feature set, fit, prediction and
    score already existed. In a dry run the computed counts are what a run would
    compute. Each count is per run: a feature set or fit two pipelines share counts
    once, and one already in ``planned`` from an earlier call of the same dry run
    counts in neither column, so the CLI's totals over several evaluations count
    shared work once. ``failed`` maps a failed pipeline's name to its error,
    ``failed_ids`` holds its id.
    """

    run: str = ""
    features_computed: int = 0
    fits_computed: int = 0
    predictions_computed: int = 0
    scores_recorded: int = 0
    features_reused: int = 0
    fits_reused: int = 0
    predictions_reused: int = 0
    scores_reused: int = 0
    failed: dict[str, str] = dataclasses.field(default_factory=dict)
    failed_ids: set[str] = dataclasses.field(default_factory=set)


PARQUET = formats.Format.PARQUET


FIT_HINT = (
    "targets and features are NaN outside the train segments; read them with "
    "dataset.column(name, train)"
)


@dataclasses.dataclass
class RunContext:
    """What every stage of one run shares."""

    ledger: Ledger
    dataset: Dataset
    targets: dict[str, Lag]
    evaluation: Evaluation
    root: pathlib.Path
    report: RunReport
    start: Callable[[], str]
    log: Callable[[str], None]
    dry: bool
    planned: set[str]
    features: dict[str, Features] = dataclasses.field(default_factory=dict)
    stages: dict[str, Stage] = dataclasses.field(default_factory=dict)
    prepared: dict[str, Any] = dataclasses.field(default_factory=dict)

    def stage(self, pipeline: Pipeline) -> Stage:
        """The run's one stage per pipeline id, so a shared member fits once."""
        if pipeline.id not in self.stages:
            self.stages[pipeline.id] = Stage(self, pipeline)
        return self.stages[pipeline.id]


Features = tuple[tuple[str, ...], pathlib.Path | None, str]
"""One computed feature set: its column names, the Parquet file they are read from
(None in a dry run, which stores nothing) and the id of the ordered columns' bytes."""


PLANNED_DAYS = 7
"""How long a dry run's feature set waits under ``planned`` for a run to use it."""


class Stage:
    """One pipeline's memoized features, fits and predictions; a blend holds one per
    member. ``run`` is what every stage of the run shares.
    """

    def __init__(self, run: RunContext, pipeline: Pipeline):
        self.run = run
        self.pipeline = pipeline
        self.name = pipeline.name or pipeline.id
        self.members = [run.stage(m) for m in pipeline.members]
        self.feature_functions = pipeline.feature_functions
        shared = {m.feature_functions for m in self.members}
        if not self.feature_functions and len(shared) == 1:
            self.feature_functions = shared.pop()
        self.shas = identity.import_shas(pipeline.functions, self.run.root)
        self.functions = {
            **stage_functions(pipeline),
            "features": self.feature_functions,
        }
        self.stage_shas = {
            stage: identity.import_shas(functions, self.run.root)
            for stage, functions in self.functions.items()
        }
        self.stage_keys = {
            stage: identity.code_keys(functions, self.run.root)
            for stage, functions in self.functions.items()
        }
        self.dists = identity.imported_dists(pipeline.functions, self.run.root)
        self.locks = {
            stage: lock_text(identity.imported_dists(functions, self.run.root))
            for stage, functions in self.functions.items()
        }
        self.env_lock = {
            "sha": identity.bytes_hash(self.locks["fit"].encode()),
            "format": formats.Format.TEXT,
        }
        self.feature_ids = [
            identity.content_hash(
                {
                    "dataset": self.run.evaluation.dataset_id,
                    "features": function,
                    "code_keys": identity.code_keys((function,), self.run.root),
                    "env_lock": identity.bytes_hash(
                        lock_text(
                            identity.imported_dists((function,), self.run.root)
                        ).encode()
                    ),
                }
            )
            for function in self.feature_functions
        ]
        self.featured: Dataset | None = None
        self.models: dict[str, Any] = {}
        self.pending: dict[str, tuple[Any, str, dict[str, Any]]] = {}
        self.fits: dict[tuple[tuple[int, int], ...], str] = {}
        self.widths: dict[str, tuple[int, ...]] = {}

    def fit(self, fold: Fold, index: int) -> str:
        """The fit id of the fold's train segments, computed once per run."""
        train: Segments = tuple((lo, hi) for lo, hi in fold.train)
        if train not in self.fits:
            self.fits[train] = self._fit(fold, index)
        return self.fits[train]

    def _fit(self, fold: Fold, index: int) -> str:
        train = [list(seg) for seg in fold.train]
        member_fits = [m.fit(fold, index) for m in self.members]
        columns_id = self._feature_columns_id()
        if self.pipeline.fit is None:
            return identity.content_hash(
                {
                    "stateless": True,
                    "members": member_fits,
                    **({"features": columns_id} if columns_id else {}),
                }
            )
        train_preds = (
            self._train_predictions(member_fits, fold, index)
            if self.members and positional(self.pipeline.fit) > 3
            else []
        )
        fit_id = identity.content_hash(
            {
                "dataset": self.run.evaluation.dataset_id,
                "pipeline": self.pipeline.fit_declaration,
                "train": train,
                "code_keys": self.stage_keys["fit"],
                "env_lock": self.env_lock["sha"],
                **({"members": member_fits} if self.members else {}),
                **({"member_predictions": train_preds} if train_preds else {}),
                **({"features": columns_id} if columns_id else {}),
            }
        )
        if fit_id in self.run.planned:
            return fit_id
        if self.run.ledger.latest(Event.FIT, fit_id) is not None:
            self.run.report.fits_reused += 1
            self.run.planned.add(fit_id)
            return fit_id
        if self.run.dry:
            self.run.planned.add(fit_id)
            self.run.report.fits_computed += 1
            self.run.log(
                f"would fit {self.name} {fold.label}: {self._why(fold.label, train)}"
            )
            return fit_id
        split = self.run.evaluation.split
        visible = FitView(
            self._visible(max(hi for _, hi in fold.train)).train_view(fold.train),
            fold.train,
            embargo_rows=getattr(split, "embargo_rows", 0),
            embargo_timestamps=getattr(split, "embargo_timestamps", 0),
        )
        if self.pipeline.prepare is not None:
            prepared = self._prepare(self.pipeline.prepare, visible, fold, columns_id)
            visible = dataclasses.replace(visible, prepared=prepared)
        inputs = (
            [
                [
                    np.concatenate([load_predictions(self.run.ledger, p) for p in ids])
                    for ids in train_preds
                ]
            ]
            if train_preds
            else []
        )
        run_id = self.run.start()
        loaded = set(sys.modules)
        why = self._why(fold.label, train)
        self.run.log(f"fitting {self.name} {fold.label}")
        started = time.perf_counter()
        model = noted(
            f"fit on fold {fold.label}; {FIT_HINT}",
            self.pipeline.fit,
            visible,
            fold.train,
            self.pipeline.params,
            *inputs,
        )
        duration = time.perf_counter() - started
        facts = None
        if isinstance(model, Fitted):
            facts = _facts(self.name, model.facts)
            model = model.model
        blob = self._model_blob(model)
        load = _declared(self.pipeline.load, self.name, "load")
        self.models[fit_id] = load(self.run.ledger.get_blob(blob["sha"]))
        self._check_imports("fit", loaded)
        self._unchanged("fit")
        self.run.ledger.put_blob(self.locks["fit"].encode())
        self.pending[fit_id] = (
            model,
            why,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "train": train,
                "label": fold.label,
                "import_shas": self.shas,
                "code_keys": self.stage_keys["fit"],
                "env_lock": self.env_lock,
                "duration_s": duration,
                "model": blob,
                **({"members": member_fits} if self.members else {}),
                **({"features": columns_id} if columns_id else {}),
                **({"facts": facts} if facts is not None else {}),
            },
        )
        self.run.report.fits_computed += 1
        return fit_id

    def predictions(
        self, fit_id: str, fold: Fold, index: int, rows: Range, train: bool = False
    ) -> str:
        """The prediction id over ``rows``: the fold's test range, or with ``train`` one
        of its train segments, seen through the fit's view, for an in-sample blend.
        """
        member_fits = [m.fit(fold, index) for m in self.members]
        member_preds = [
            m.predictions(f, fold, index, rows, train)
            for m, f in zip(self.members, member_fits, strict=True)
        ]
        raw_id = identity.content_hash(
            {
                "fit": fit_id,
                "range": list(rows),
                "predict": self.pipeline.predict,
                "code_keys": self.stage_keys["predict"],
                "env_lock": identity.bytes_hash(self.locks["predict"].encode()),
                **({"members": member_preds} if self.members else {}),
                **(
                    {"predict_config": self.pipeline.predict_config}
                    if self.pipeline.predict_config is not None
                    else {}
                ),
            }
        )
        pred_id = raw_id
        if self.pipeline.postprocess is not None:
            pred_id = identity.content_hash(
                {
                    "raw": raw_id,
                    "postprocess": self.pipeline.postprocess,
                    "postprocess_config": self.pipeline.postprocess_config,
                    "code_keys": self.stage_keys["postprocess"],
                    "env_lock": identity.bytes_hash(self.locks["postprocess"].encode()),
                }
            )
        if pred_id in self.run.planned:
            return pred_id
        if self.run.ledger.latest(Event.PREDICTIONS, pred_id):
            self.run.report.predictions_reused += 1
            self.run.planned.add(pred_id)
            return pred_id
        if self.run.dry:
            raw_missing = raw_id not in self.run.planned and not self.run.ledger.latest(
                Event.PREDICTIONS, raw_id
            )
            self.run.report.predictions_computed += 1 + (
                raw_missing and raw_id != pred_id
            )
            self.run.planned.update((raw_id, pred_id))
            return pred_id
        config = self.pipeline.predict_config
        where: dict[str, Any] = {
            "fit": fit_id,
            "range": list(rows),
            "fold": index,
            **({"members": member_preds} if self.members else {}),
            **(
                {"predict_config": identity.canonical(config)}
                if config is not None
                else {}
            ),
        }
        visible = self._visible(rows[1])
        if train:
            visible = visible.train_view(fold.train)
        visible = visible.masked(self.run.targets, rows[0])
        self.run.start()
        if self.run.ledger.latest(Event.PREDICTIONS, raw_id) is not None:
            raw = load_predictions(self.run.ledger, raw_id)
        else:
            inputs = [load_predictions(self.run.ledger, p) for p in member_preds]

            def predict(model: Any, view: Dataset, part: Range) -> np.ndarray:
                head = part[1] - rows[0]
                extra = [[a[:head] for a in inputs]] if self.members else []
                if config is not None:
                    extra.append(config)
                return np.asarray(
                    noted(
                        f"predict on fold {fold.label}",
                        self.pipeline.predict,
                        model,
                        view,
                        part,
                        *extra,
                    )
                )

            loaded = set(sys.modules)
            model = self._model(fit_id)
            self._unchanged("predict")
            raw = self._checked_rows("predict", predict(model, visible, rows), rows)
            if fit_id in self.pending:
                self._record_fit(fit_id, fold, raw, lambda m: predict(m, visible, rows))
            self._probe(
                "predict",
                raw,
                lambda v, part: predict(self._model(fit_id, fresh=True), v, part),
                visible,
                rows,
            )
            self._check_imports("predict", loaded)
            if self.pipeline.postprocess is None and not train:
                self._refuse_non_finite("predict", raw, fold, rows)
            raw = self._write(raw_id, raw, where)
            self.run.log(f"predictions {self.name} rows {rows[0]}:{rows[1]}")
        if self.pipeline.postprocess is not None:
            config = self.pipeline.postprocess_config
            postprocess = functools.partial(
                noted, f"postprocess on fold {fold.label}", self.pipeline.postprocess
            )
            self._unchanged("postprocess")
            loaded = set(sys.modules)
            post = self._checked_rows(
                "postprocess", postprocess(raw, visible, rows, config), rows
            )
            self._probe(
                "postprocess",
                post,
                lambda view, part: postprocess(
                    raw[: part[1] - rows[0]], view, part, config
                ),
                visible,
                rows,
            )
            self._check_imports("postprocess", loaded)
            if not train:
                self._refuse_non_finite("postprocess", post, fold, rows)
            self._write(
                pred_id,
                post,
                {
                    **where,
                    "raw": raw_id,
                    "postprocess": identity.canonical(self.pipeline.postprocess),
                    "postprocess_config": identity.canonical(config),
                },
            )
            self.run.log(f"postprocess {self.name} rows {rows[0]}:{rows[1]}")
        return pred_id

    def _prepare(
        self, prepare: Callable, visible: FitView, fold: Fold, columns_id: str | None
    ) -> Any:
        """What ``prepare`` builds for the fold's train range.

        Built once per run across every pipeline declaring the same function, code,
        environment, dataset, feature columns and train segments.
        """
        if not hasattr(self, "prepare_key"):
            self.prepare_key = {
                "prepare": prepare,
                "code_keys": identity.code_keys((prepare,), self.run.root),
                "env_lock": identity.bytes_hash(
                    lock_text(
                        identity.imported_dists((prepare,), self.run.root)
                    ).encode()
                ),
            }
        key = identity.content_hash(
            {
                **self.prepare_key,
                "dataset": self.run.evaluation.dataset_id,
                "train": [list(seg) for seg in fold.train],
                **({"features": columns_id} if columns_id else {}),
            }
        )
        if key not in self.run.prepared:
            started = time.perf_counter()
            self.run.prepared[key] = frozen(
                noted(f"prepare on fold {fold.label}", prepare, visible, fold.train)
            )
            duration = time.perf_counter() - started
            self.run.log(f"prepare {self.name} {fold.label} {duration:.1f}s")
        return self.run.prepared[key]

    def _record_fit(
        self, fit_id: str, fold: Fold, loaded: np.ndarray, predict: Callable
    ):
        """Append a fit once its reloaded model predicts its first range as the
        fitted one did, so a save that drops state is refused, not recorded.
        """
        fitted, why, payload = self.pending.pop(fit_id)
        again = np.asarray(predict(fitted))
        refuse_changed(f"{self.name}: fit {fold.label}", "prediction", again, loaded)
        self.run.ledger.append(
            Event.FIT, self.run.evaluation.dataset_id, fit_id, payload, id=fit_id
        )
        self.run.log(
            f"fit {self.name} {fold.label} {payload['duration_s']:.1f}s: {why}"
        )

    def _train_predictions(
        self, member_fits: list[str], fold: Fold, index: int
    ) -> list[list[str]]:
        return [
            [member.predictions(fit_id, fold, index, seg, True) for seg in fold.train]
            for member, fit_id in zip(self.members, member_fits, strict=True)
        ]

    def _unchanged(self, stage: str):
        """Refuse to record a sha for code that is not the code that ran."""
        now = identity.import_shas(self.functions[stage], self.run.root)
        moved = sorted(
            k
            for k in set(now) | set(self.stage_shas[stage])
            if now.get(k) != self.stage_shas[stage].get(k)
        )
        if moved:
            raise Refused(f"{self.name}: source changed during the run: {moved}; rerun")

    def _visible(self, cut: int) -> Dataset:
        """The dataset's first ``cut`` rows, with the pipeline's feature columns read
        on demand from their stored files.
        """
        if not self.feature_functions:
            return self.run.dataset.upto(cut)
        if self.featured is None:
            sets = [
                self._features(f, i)
                for f, i in zip(self.feature_functions, self.feature_ids, strict=True)
            ]
            names = tuple(n for columns, _, _ in sets for n in columns)
            stores = {
                n: path
                for columns, path, _ in sets
                for n in columns
                if path is not None
            }
            self.featured = Dataset(
                self.run.dataset.columns,
                self.run.dataset.ts,
                names,
                stores,
                reveal=self.run.dataset.reveal,
            )
        return self.featured.upto(cut)

    def _feature_columns_id(self) -> str | None:
        """The fit's feature input: the id of the ordered columns' bytes, computing the
        features first when no run has, so a dry run's ids are the real ones.
        """
        sets = [
            self._features(f, i)
            for f, i in zip(self.feature_functions, self.feature_ids, strict=True)
        ]
        names = Counter(n for columns, _, _ in sets for n in columns)
        twice = sorted(n for n, k in names.items() if k > 1)
        if twice:
            raise Refused(
                f"{self.name}: features functions name the same columns {twice}"
            )
        return identity.content_hash([i for _, _, i in sets]) if sets else None

    def _model_blob(self, model: Any) -> dict[str, Any]:
        """The saved model; a zip is portable iff every member's format is."""
        payload = _declared(self.pipeline.save, self.name, "save")(model)
        used = {"model": _declared(self.pipeline.format, self.name, "format")}
        if self.pipeline.format == formats.Format.ZIP:
            used = formats.zip_formats(payload)
            bad = sorted(
                f for f in used.values() if f not in formats.KNOWN or f == "zip"
            )
            if bad:
                raise Refused(
                    f"{self.name}: zip members declare formats {bad}; each must be "
                    "one of formats.KNOWN other than zip"
                )
        return {
            "sha": self.run.ledger.put_blob(payload),
            "format": self.pipeline.format,
            "portable": all(formats.KNOWN[f] for f in used.values()),
        }

    def _features(self, function: Callable, feature_id: str) -> Features:
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
            found = (tuple(payload["columns"]), path, payload["columns_id"])
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
        planned = self._planned_features(feature_id)
        if planned is not None and self.run.dry:
            columns_id, columns, duration = planned
            self.run.log(
                f"would compute features {self.name} {ref} {len(columns)} columns: "
                f"computed and probed by an earlier dry run"
            )
            self.run.report.features_computed += 1
            self.run.planned.add(feature_id)
            found = (tuple(columns), None, columns_id)
            _PLANNED_FEATURES[feature_id] = found
            return self.run.features.setdefault(feature_id, found)
        loaded = set(sys.modules)
        if planned is None:
            self.run.log(f"computing features {self.name} {ref}")
            started = time.perf_counter()
            columns = self._feature_columns(function, bare)
            duration = time.perf_counter() - started
        else:
            self.run.log(f"features {self.name} {ref}: from the dry run, probing")
            columns_id, columns, duration = planned
        columns_id = columns_hash(columns)
        probe_rows = self._probe_features(
            function, bare, revealed, columns, spans=1 if self.run.dry else 5
        )
        self._check_imports("features", loaded)
        self._unchanged("features")
        if self.run.dry:
            self.run.log(
                f"would compute features {self.name} {ref} {len(columns)} columns "
                f"{duration:.1f}s: {self._features_why(function)}; probed at rows "
                f"{probe_rows} only"
            )
            self._plan_features(feature_id, columns, columns_id, duration)
            self.run.report.features_computed += 1
            self.run.planned.add(feature_id)
            found = (tuple(columns), None, columns_id)
            _PLANNED_FEATURES[feature_id] = found
            return self.run.features.setdefault(feature_id, found)
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
                "blob": {"sha": sha, "format": PARQUET},
            },
            id=feature_id,
        )
        self.run.report.features_computed += 1
        self.run.log(f"features {self.name} {len(columns)} columns {duration:.1f}s")
        for leftover in self._planned_dir().glob(f"{feature_id}.*"):
            leftover.unlink(missing_ok=True)
        return self.run.features.setdefault(
            feature_id, (tuple(columns), self.run.ledger.blobs / sha, columns_id)
        )

    def _planned_dir(self) -> pathlib.Path:
        return self.run.ledger.root / "planned"

    def _plan_features(
        self,
        feature_id: str,
        columns: dict[str, np.ndarray],
        columns_id: str,
        duration: float,
    ):
        """Keep a dry run's feature set under ``planned``, outside the events and the
        blob store, so the run that follows reads it instead of computing it again;
        sets older than ``PLANNED_DAYS`` are dropped, since no run came for them.
        """
        planned = self._planned_dir()
        planned.mkdir(exist_ok=True)
        stale = time.time() - PLANNED_DAYS * 86400
        for old in planned.iterdir():
            try:
                if old.stat().st_mtime < stale:
                    old.unlink(missing_ok=True)
            except OSError:
                continue
        (planned / f"{feature_id}.parquet").write_bytes(
            rows_save(Dataset(columns, None))
        )
        (planned / f"{feature_id}.json").write_text(
            json.dumps(
                {
                    "columns": list(columns),
                    "columns_id": columns_id,
                    "duration_s": duration,
                }
            )
        )

    def _planned_features(
        self, feature_id: str
    ) -> tuple[str, dict[str, np.ndarray], float] | None:
        """A feature set an earlier dry run left under ``planned``, when its columns
        still hash to the id it was saved with; None otherwise.
        """
        meta = self._planned_dir() / f"{feature_id}.json"
        data = self._planned_dir() / f"{feature_id}.parquet"
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

    def _probe_features(
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
            head = self._feature_columns(
                function, bare.upto(row).masked(revealed, row - 1)
            )
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
                    plain = self._feature_columns(function, bare.upto(row))[name]
                    if np.array_equal(plain, values[:row], equal_nan=True):
                        raise Refused(
                            f"{self.name}: features read a revealed target before its "
                            f"reveal lag: column {name} changes before row {row} when "
                            f"labels not yet known at row {row - 1} are hidden"
                        ) from error
                    raise
            del head
        return probe_rows

    def _features_why(self, function: Callable) -> str:
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

    def _feature_columns(
        self, function: Callable, bare: Dataset
    ) -> dict[str, np.ndarray]:
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

    def _why(self, label: str, train: list[list[int]]) -> str:
        """Why a fit is computed: the inputs that moved against the nearest earlier fit
        of this pipeline, by id or name, and label, the one with the fewest moved, the
        newest among ties.
        """
        rows = self.run.ledger.sql(
            "SELECT f.dataset, f.train, f.code_keys, f.env_lock, (SELECT d.payload "
            "FROM event d WHERE d.type = 'pipeline_declared' AND d.key = f.pipeline "
            "LIMIT 1) AS declared FROM event_fit f JOIN event_pipeline p "
            "ON p.id = f.pipeline WHERE (f.pipeline = ? OR p.name = ?) "
            "AND f.label = ? ORDER BY f.seq DESC",
            (self.pipeline.id, self.pipeline.name, label),
        )
        if not rows:
            return "no earlier fit of this pipeline and label"
        now = self.stage_keys["fit"]
        params = identity.content_hash(self.pipeline.params)

        def moved(row: dict[str, Any]) -> list[str]:
            before = json.loads(row["code_keys"])
            files = sorted(
                k for k in set(before) | set(now) if before.get(k) != now.get(k)
            )
            declared = json.loads(row["declared"])["params"]
            return [
                reason
                for reason, differs in (
                    (
                        "dataset changed",
                        row["dataset"] != self.run.evaluation.dataset_id,
                    ),
                    ("params changed", identity.content_hash(declared) != params),
                    (f"code changed: {', '.join(files)}", bool(files)),
                    (
                        "environment lock changed",
                        json.loads(row["env_lock"])["sha"] != self.env_lock["sha"],
                    ),
                    ("train segments changed", json.loads(row["train"]) != train),
                )
                if differs
            ]

        nearest = min((moved(r) for r in rows), key=len)
        return "; ".join(nearest) or "members or feature columns changed"

    def _probe(
        self,
        stage: str,
        output: np.ndarray,
        compute: Callable[[Dataset, Range], Any],
        visible: Dataset,
        rows: Range,
    ):
        """Rerun the function on the rows before each cut of ``cuts``, with its array
        inputs cut there, and require the predictions before it to stand; every computed
        range, before it is written.
        """
        lo, hi = rows
        for at in cuts(visible, lo, hi):
            again = np.asarray(compute(visible.upto(at), (lo, at)))
            refuse_changed(
                f"{self.name}: {stage}", "prediction", output, again, at=at, first=lo
            )

    def _checked_rows(self, stage: str, output: Any, rows: Range) -> np.ndarray:
        values = one_per_row(f"{self.name}: {stage} returned", output, rows)
        n = rows[1] - rows[0]
        width = self.widths.setdefault(stage, values.shape[1:])
        if values.shape[1:] != width:
            raise Refused(
                f"{self.name}: {stage} returned shape {values.shape} at rows "
                f"{rows[0]}:{rows[1]} and shape ({n}, {', '.join(map(str, width))}) "
                f"on an earlier fold; the width is a property of {stage}"
            )
        return values

    def _refuse_non_finite(
        self, stage: str, values: np.ndarray, fold: Fold, rows: Range
    ):
        bad = ~np.isfinite(np.asarray(values, np.float64)).reshape(len(values), -1)
        bad = np.flatnonzero(bad.any(axis=1))
        if bad.size:
            raise Refused(
                f"{self.name}: {stage} returned a non-finite value on {bad.size} of "
                f"{len(values)} rows of fold {fold.label}, the first at row "
                f"{rows[0] + bad[0]}; a scored prediction is finite, so fill or drop "
                f"them in {stage}"
            )

    def _check_imports(self, stage: str, loaded: set[str]):
        """Check each computation of a stage before its write, so an import on a later
        fold is named as one.
        """
        refuse_lazy_imports(self.pipeline, self.run.root, self.dists, loaded, stage)

    def _model(self, fit_id: str, fresh: bool = False) -> Any:
        """The fit's model loaded from its saved bytes once per run, or with ``fresh`` a
        new load, so a probe meets no state an earlier predict left in the model.
        """
        if self.pipeline.fit is None:
            return None
        if fresh or fit_id not in self.models:
            fit = self.run.ledger.event(Event.FIT, fit_id)
            load = _declared(self.pipeline.load, self.name, "load")
            model = load(self.run.ledger.get_blob(fit["payload"]["model"]["sha"]))
            if fresh:
                return model
            self.models[fit_id] = model
        return self.models[fit_id]

    def _write(self, pred_id: str, pred: Any, payload: dict[str, Any]) -> np.ndarray:
        values = np.asarray(pred, dtype=np.float64)
        blob = {
            "sha": self.run.ledger.put_blob(formats.series_save(values)),
            "format": PARQUET,
        }
        self.run.ledger.append(
            Event.PREDICTIONS,
            self.run.evaluation.dataset_id,
            pred_id,
            {**payload, "blob": blob},
            id=pred_id,
        )
        self.run.report.predictions_computed += 1
        return values


def columns_hash(columns: dict[str, np.ndarray]) -> str:
    """The ordered columns' names, dtypes and bytes: what a fit consumes."""
    return identity.content_hash(
        [[n, str(v.dtype), identity.array_hash(v)] for n, v in columns.items()]
    )


def load_predictions(ledger: Ledger, pred_id: str) -> np.ndarray:
    event = ledger.event(Event.PREDICTIONS, pred_id)
    return formats.series_load(ledger.get_blob(event["payload"]["blob"]["sha"]))


# Private Functions ==================================================================


_PLANNED_FEATURES: dict[str, Features] = {}
"""Feature sets a dry run computed in this process, by feature id, so a later dry run
of the same features under another evaluation reuses the names and id instead of
computing them again. The id covers the dataset, the function, its code and its
environment, so an entry is right for as long as the process runs. On disk the same
sets live under the ledger's ``planned`` directory (``_planned_dir``), where a later
process, dry or real, picks them up."""


def _facts(name: str, facts: Any) -> dict[str, Any]:
    """A fit's facts as the log stores them: a dict of JSON values."""
    if not isinstance(facts, dict):
        raise Refused(
            f"{name}: fit returned Fitted with facts of type "
            f"{type(facts).__name__}; facts are a dict of JSON values"
        )
    try:
        return json.loads(json.dumps(facts, default=_fact))
    except (TypeError, ValueError, Refused) as error:
        raise Refused(f"{name}: fit facts are not JSON values: {error}") from error


def _fact(value: Any) -> Any:
    """A fact as JSON.

    An array or numpy scalar by its values, anything else as a declaration
    canonicalises it.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return identity.canonical(value)


def _declared[T](value: T | None, name: str, slot: str) -> T:
    """A slot of a fitting pipeline, which ``refuse_half_declared`` made sure of."""
    if value is None:
        raise Refused(f"pipeline {name}: {slot} is not declared")
    return value
