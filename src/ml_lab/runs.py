"""Runs: split a dataset, memoize fits and predictions, record a score per pipeline.

A run owns the folds, the clock, the guards and the scorer. A pipeline's functions see
the rows their slot may read, and the scorer's ``series`` sees the fold's test range
with its predictions. Reading is SQL over the views in ``ml_lab.ledger``.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> report.fits_computed, report.scores_recorded  # doctest: +SKIP
(9, 1)
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import inspect
import json
import os
import pathlib
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ml_lab import formats, identity
from ml_lab.dataset import Dataset, Lag, Range, load, resolve, row_mask, rows_save
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.splits import Fold


@dataclasses.dataclass
class RunReport:
    """What a run wrote; ``run`` is empty when every fit, prediction and score already
    existed. In a dry run the computed counts are what a run would compute.
    ``failed`` maps a failed pipeline's name to its error, ``failed_ids`` holds its id.
    """

    run: str = ""
    fits_computed: int = 0
    predictions_computed: int = 0
    scores_recorded: int = 0
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


# Public Functions =====================================================================


def run(
    ledger: Ledger,
    pipelines: list[Pipeline],
    evaluation: Evaluation,
    *,
    code_root: pathlib.Path | None = None,
    log: Callable[[str], None] = lambda line: None,
    dry: bool = False,
    planned: set[str] | None = None,
) -> RunReport:
    """Fit, predict and score each pipeline, reusing what the memo rule allows.

    ``run_started`` is appended only when something is computed, right before the first
    write, so a rerun of an unchanged tree writes nothing and adding one pipeline costs
    only its own fits, predictions and score. Every function sees a prefix of the
    dataset: ``fit`` up to the end of its last train segment with every column NaN
    outside the train segments, ``predict``, ``postprocess`` and the scorer's
    ``series`` up to the end of the fold's test range with unknown labels NaN. ``log``
    receives one line per feature set, fit, prediction set and score as it is written.
    A ``dry`` run computes every id, looks each one up and writes nothing; it logs what
    it would compute and, for a fit, which repo files moved against the newest earlier
    fit of the same pipeline and label. ``planned`` carries the ids a dry run would
    write across several calls, so shared work counts once. ``code_root`` is where
    repo modules are hashed from, the git root of the first pipeline's fit when not
    given.

    Refused before anything is read or written, at any depth of blends inside blends:
    no pipelines; two names on one declaration; a slot holding a lambda, closure,
    method or anything that is not a module-level function; a ``format`` outside
    ``formats.KNOWN``; a params class at any depth defined in a repo module its
    function does not import; a reached module the memo cannot see
    (``identity.refuse_unseen_code``).

    Refused before anything is written: a ``dataset_id`` prefix that matches no
    recorded dataset or several; a split's own refusals (too few folds, an empty test
    range, a fold label twice, a group that recurs); a fold whose test range overlaps
    its train segments,
    whose folds train on labels not revealed when their test range starts (the message
    names the purge that clears them); a split whose folds moved under an unchanged
    declaration; a sealed evaluation on a dataset without a sealed tail, an
    evaluation that is not sealed validating on one, a sealed evaluation validating
    before it, and a fold training on a tail row no earlier fold of a sealed
    evaluation tested.

    Recorded as ``pipeline_failed``, after which the others continue and a rerun resumes
    from what was written: an error a function raises, noted with the fold's label; an
    output that is not one row per row of the range, or whose width changes between
    folds; a prediction or features column that changes when recomputed on a shorter
    prefix; a model reloaded from its saved bytes that predicts differently; a scored
    prediction or a metric that is not finite; a series that is a scalar or empty, or
    whose column count ``Scorer.columns`` disagrees with; an import inside a function
    body; a module edited during the run; a second score of one pipeline id under a
    sealed evaluation, found by computing its ids before any fit or prediction, and
    held at the write against a concurrent run.
    """
    if not pipelines:
        raise Refused("no pipelines declared")
    names: dict[str, str] = {}
    for pipeline in pipelines:
        name = pipeline.name or pipeline.id
        if names.setdefault(pipeline.id, name) != name:
            raise Refused(
                f"pipelines {names[pipeline.id]} and {name} declare the same functions "
                f"and params, one id {pipeline.id}; rename or change one"
            )
        for p in _nested(pipeline):
            for slot, held in [(s, getattr(p, s)) for s in _SLOTS] + [
                ("features", f) for f in p.feature_functions
            ]:
                if held is not None and not callable(held):
                    raise Refused(
                        f"pipeline {p.name or name}: {slot} holds a "
                        f"{type(held).__name__}, not a function"
                    )
        if pipeline.format not in formats.KNOWN:
            raise Refused(
                f"pipeline {name}: format {pipeline.format!r}, not one of "
                f"{sorted(formats.KNOWN)}"
            )
    root = code_root or identity.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
    scoring = _scoring(evaluation)
    identity.refuse_unseen_code(
        [
            *scoring,
            *(
                f
                for pipeline in pipelines
                for p in _nested(pipeline)
                for stage in _stages(p).values()
                for f in stage
            ),
        ],
        root,
    )
    for pipeline in pipelines:
        for p in _nested(pipeline):
            shas = identity.import_shas((p.fit, p.save, p.load), root)
            _refuse_unseen_class(p.params, shas, root, "fit")
            if p.postprocess is not None:
                shas = identity.import_shas((p.postprocess,), root)
                _refuse_unseen_class(p.postprocess_config, shas, root, "postprocess")
    scorer_shas = identity.import_shas(scoring[:2], root)
    _refuse_unseen_class(evaluation.scorer_params, scorer_shas, root, "scorer")
    evaluation = dataclasses.replace(
        evaluation, dataset_id=resolve(ledger, evaluation.dataset_id)
    )
    dataset = load(ledger, evaluation.dataset_id)
    recorded = ledger.latest(Event.DATASET, evaluation.dataset_id)["payload"]
    targets = {name: _lag(lag) for name, lag in recorded["recipe"]["targets"].items()}
    schedule = evaluation.split.folds(dataset)
    _refuse_overlap(schedule)
    _refuse_unrevealed(dataset, targets, schedule)
    _refuse_sealed(evaluation, schedule, recorded["sealed_from"])
    _declare_evaluation(ledger, evaluation, schedule, dry)
    report = RunReport()

    def start() -> str:
        if not report.run:
            run_id = identity.ulid()
            ledger.append(
                Event.RUN,
                evaluation.id,
                run_id,
                {
                    "git": _git(ledger, root),
                    "resolution": _resolution(ledger, root),
                    "editable": {
                        name: _git(ledger, at)
                        for name, at in identity.editable_roots(
                            [*scoring, *(f for p in pipelines for f in p.functions)],
                            root,
                        ).items()
                    },
                    "host": _host(),
                    "pipelines": [{"id": p.id, "name": p.name} for p in pipelines],
                },
                id=run_id,
            )
            report.run = run_id
        return report.run

    context = _Run(
        ledger,
        dataset,
        targets,
        evaluation,
        root,
        report,
        start,
        log,
        dry,
        set() if planned is None else planned,
    )
    for pipeline in pipelines:
        if not dry:
            for declared in _nested(pipeline):
                _declare_pipeline(ledger, declared)
        try:
            if (
                evaluation.sealed
                and not dry
                and _held_score(ledger, evaluation, pipeline)
            ):
                quiet = dataclasses.replace(
                    context,
                    report=RunReport(),
                    log=lambda line: None,
                    dry=True,
                    planned=set(),
                    features={},
                    stages={},
                )
                _run_pipeline(quiet, schedule, pipeline)
            _run_pipeline(context, schedule, pipeline)
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = "; ".join(
                [f"{type(error).__name__}: {error}", *getattr(error, "__notes__", ())]
            )
            report.failed_ids.add(pipeline.id)
            context.stages.clear()
            if dry:
                continue
            ledger.append(
                Event.FAILED,
                evaluation.id,
                pipeline.id,
                {
                    "run": start(),
                    "pipeline": pipeline.id,
                    "error": report.failed[name],
                    "traceback": _text_blob(ledger, traceback.format_exc()),
                },
            )
    return report


# Running one pipeline =================================================================


@dataclasses.dataclass
class _Run:
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
    stages: dict[str, _Stage] = dataclasses.field(default_factory=dict)

    def stage(self, pipeline: Pipeline) -> _Stage:
        """The run's one stage per pipeline id, so a shared member fits once."""
        if pipeline.id not in self.stages:
            self.stages[pipeline.id] = _Stage(self, pipeline)
        return self.stages[pipeline.id]


Features = tuple[tuple[str, ...], pathlib.Path | None, str]
"""One computed feature set: its column names, the Parquet file they are read from
(None in a dry run, which stores nothing) and the id of the ordered columns' bytes."""


def _run_pipeline(context: _Run, folds: list[Fold], pipeline: Pipeline):
    stage = context.stage(pipeline)
    predictions: dict[str, str] = {}
    for index, fold in enumerate(folds):
        fit_id = stage.fit(fold, index)
        predictions[str(index)] = stage.predictions(fit_id, fold, index, fold.test)
    _score(context, folds, pipeline, predictions)


class _Stage:
    """One pipeline's memoized features, fits and predictions; a blend holds one per
    member.
    """

    def __init__(self, context: _Run, pipeline: Pipeline):
        vars(self).update(vars(context))
        self.pipeline = pipeline
        self.name = pipeline.name or pipeline.id
        self.members = [context.stage(m) for m in pipeline.members]
        self.feature_functions = pipeline.feature_functions
        shared = {m.feature_functions for m in self.members}
        if not self.feature_functions and len(shared) == 1:
            self.feature_functions = shared.pop()
        self.shas = identity.import_shas(pipeline.functions, self.root)
        self.functions = {
            **_stages(pipeline),
            "features": self.feature_functions,
        }
        self.stage_shas = {
            stage: identity.import_shas(functions, self.root)
            for stage, functions in self.functions.items()
        }
        self.stage_keys = {
            stage: identity.code_keys(functions, self.root)
            for stage, functions in self.functions.items()
        }
        self.dists = identity.imported_dists(pipeline.functions, self.root)
        self.locks = {
            stage: _lock_text(identity.imported_dists(functions, self.root))
            for stage, functions in self.functions.items()
        }
        self.env_lock = {
            "sha": identity.bytes_hash(self.locks["fit"].encode()),
            "format": formats.Format.TEXT,
        }
        self.feature_ids = [
            identity.content_hash(
                {
                    "dataset": self.evaluation.dataset_id,
                    "features": function,
                    "code_keys": identity.code_keys((function,), self.root),
                    "env_lock": identity.bytes_hash(
                        _lock_text(
                            identity.imported_dists((function,), self.root)
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
        train = tuple(tuple(segment) for segment in fold.train)
        if train not in self.fits:
            self.fits[train] = self._fit(fold, index)
        return self.fits[train]

    def _fit(self, fold: Fold, index: int) -> str:
        train = [list(seg) for seg in fold.train]
        member_fits = [m.fit(fold, index) for m in self.members]
        train_preds = (
            self._train_predictions(member_fits, fold, index)
            if self.members and _positional(self.pipeline.fit) > 3
            else []
        )
        columns_id = self._feature_columns_id()
        fit_id = identity.content_hash(
            {
                "dataset": self.evaluation.dataset_id,
                "pipeline": self.pipeline.fit_declaration,
                "train": train,
                "code_keys": self.stage_keys["fit"],
                "env_lock": self.env_lock["sha"],
                **({"members": member_fits} if self.members else {}),
                **({"member_predictions": train_preds} if train_preds else {}),
                **({"features": columns_id} if columns_id else {}),
            }
        )
        if fit_id in self.planned:
            return fit_id
        if self.ledger.latest(Event.FIT, fit_id) is not None:
            self.report.fits_reused += 1
            self.planned.add(fit_id)
            return fit_id
        if self.dry:
            self.planned.add(fit_id)
            self.report.fits_computed += 1
            self.log(f"would fit {self.name} {fold.label}: {self._why(fold.label)}")
            return fit_id
        visible = self._visible(max(hi for _, hi in fold.train)).train_view(fold.train)
        inputs = (
            [
                [
                    np.concatenate([_load_predictions(self.ledger, p) for p in ids])
                    for ids in train_preds
                ]
            ]
            if train_preds
            else []
        )
        run_id = self.start()
        loaded = set(sys.modules)
        why = self._why(fold.label)
        started = time.perf_counter()
        model = _noted(
            f"fit on fold {fold.label}; {FIT_HINT}",
            self.pipeline.fit,
            visible,
            fold.train,
            self.pipeline.params,
            *inputs,
        )
        duration = time.perf_counter() - started
        blob = self._model_blob(model)
        self.models[fit_id] = self.pipeline.load(self.ledger.get_blob(blob["sha"]))
        self._check_imports("fit", loaded)
        self._unchanged("fit")
        self.ledger.put_blob(self.locks["fit"].encode())
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
            },
        )
        self.report.fits_computed += 1
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
        if pred_id in self.planned:
            return pred_id
        if self.ledger.latest(Event.PREDICTIONS, pred_id):
            self.report.predictions_reused += 1
            self.planned.add(pred_id)
            return pred_id
        if self.dry:
            raw_missing = raw_id not in self.planned and not self.ledger.latest(
                Event.PREDICTIONS, raw_id
            )
            self.report.predictions_computed += 1 + (raw_missing and raw_id != pred_id)
            self.planned.update((raw_id, pred_id))
            return pred_id
        where: dict[str, Any] = {
            "fit": fit_id,
            "range": list(rows),
            "fold": index,
            **({"members": member_preds} if self.members else {}),
        }
        visible = self._visible(rows[1])
        if train:
            visible = visible.train_view(fold.train)
        visible = visible.masked(self.targets, rows[0])
        self.start()
        if self.ledger.latest(Event.PREDICTIONS, raw_id) is not None:
            raw = _load_predictions(self.ledger, raw_id)
        else:
            inputs = [_load_predictions(self.ledger, p) for p in member_preds]

            def predict(model: Any, view: Dataset, part: Range) -> np.ndarray:
                head = part[1] - rows[0]
                extra = [[a[:head] for a in inputs]] if self.members else []
                return np.asarray(
                    _noted(
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
            raw = self._one_per_row("predict", predict(model, visible, rows), rows)
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
            self.log(f"predictions {self.name} rows {rows[0]}:{rows[1]}")
        if self.pipeline.postprocess is not None:
            config = self.pipeline.postprocess_config
            postprocess = functools.partial(
                _noted, f"postprocess on fold {fold.label}", self.pipeline.postprocess
            )
            self._unchanged("postprocess")
            loaded = set(sys.modules)
            post = self._one_per_row(
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
            self.log(f"postprocess {self.name} rows {rows[0]}:{rows[1]}")
        return pred_id

    def _record_fit(
        self, fit_id: str, fold: Fold, loaded: np.ndarray, predict: Callable
    ):
        """Append a fit once its reloaded model predicts its first range as the
        fitted one did, so a save that drops state is refused, not recorded.
        """
        fitted, why, payload = self.pending.pop(fit_id)
        again = np.asarray(predict(fitted))
        _refuse_changed(f"{self.name}: fit {fold.label}", "prediction", again, loaded)
        self.ledger.append(
            Event.FIT, self.evaluation.dataset_id, fit_id, payload, id=fit_id
        )
        self.log(f"fit {self.name} {fold.label} {payload['duration_s']:.1f}s: {why}")

    def _train_predictions(
        self, member_fits: list[str], fold: Fold, index: int
    ) -> list[list[str]]:
        return [
            [member.predictions(fit_id, fold, index, seg, True) for seg in fold.train]
            for member, fit_id in zip(self.members, member_fits, strict=True)
        ]

    def _unchanged(self, stage: str):
        """Refuse to record a sha for code that is not the code that ran."""
        now = identity.import_shas(self.functions[stage], self.root)
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
            return self.dataset.upto(cut)
        if self.featured is None:
            sets = [
                self._features(f, i)
                for f, i in zip(self.feature_functions, self.feature_ids, strict=True)
            ]
            names = tuple(n for columns, _, _ in sets for n in columns)
            stores = {n: path for columns, path, _ in sets for n in columns}
            self.featured = Dataset(
                self.dataset.columns, self.dataset.ts, names, stores
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
        payload = self.pipeline.save(model)
        used = {"model": self.pipeline.format}
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
            "sha": self.ledger.put_blob(payload),
            "format": self.pipeline.format,
            "portable": all(formats.KNOWN[f] for f in used.values()),
        }

    def _features(self, function: Callable, feature_id: str) -> Features:
        """One features function's columns, computed once per feature id over the
        dataset without its targets, probed and stored; a dry run computes and stores
        nothing, so the names and id are known. The probe cuts each of the five spans
        between the sixths of the rows and the end by ``_cuts``.
        """
        if feature_id in self.features:
            return self.features[feature_id]
        event = self.ledger.latest(Event.FEATURES, feature_id)
        if event is not None:
            payload = event["payload"]
            path = self.ledger.blobs / payload["blob"]["sha"]
            found = (tuple(payload["columns"]), path, payload["columns_id"])
            return self.features.setdefault(feature_id, found)
        revealed = {k: v for k, v in self.targets.items() if v is not None}
        bare = Dataset(
            {
                k: v
                for k, v in self.dataset.columns.items()
                if k in revealed or k not in self.targets
            },
            self.dataset.ts,
        )
        loaded = set(sys.modules)
        started = time.perf_counter()
        columns = self._feature_columns(function, bare)
        duration = time.perf_counter() - started
        self._check_imports("features", loaded)
        self._unchanged("features")
        columns_id = _columns_id(columns)
        if self.dry:
            return self.features.setdefault(
                feature_id, (tuple(columns), None, columns_id)
            )
        n = bare.rows
        sixths = [j * n // 6 for j in range(1, 7)]
        probe_rows = [
            at
            for span in zip(sixths, sixths[1:], strict=False)
            for at in _cuts(bare, *span)
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
                    _refuse_changed(
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
        run_id = self.start()
        sha = self.ledger.put_blob(rows_save(Dataset(columns, None)))
        self.ledger.append(
            Event.FEATURES,
            self.evaluation.dataset_id,
            feature_id,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "features": identity.canonical(function),
                "import_shas": identity.import_shas((function,), self.root),
                "code_keys": identity.code_keys((function,), self.root),
                "columns": list(columns),
                "columns_id": columns_id,
                "probe_rows": probe_rows,
                "duration_s": duration,
                "blob": {"sha": sha, "format": PARQUET},
            },
            id=feature_id,
        )
        self.log(f"features {self.name} {len(columns)} columns {duration:.1f}s")
        return self.features.setdefault(
            feature_id, (tuple(columns), self.ledger.blobs / sha, columns_id)
        )

    def _feature_columns(
        self, function: Callable, bare: Dataset
    ) -> dict[str, np.ndarray]:
        out = function(bare)
        clash = sorted(set(out) & set(self.dataset.columns))
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

    def _why(self, label: str) -> str:
        """Why a dry run would fit: the repo files and lock that moved against the
        newest earlier fit of this pipeline and label.
        """
        rows = self.ledger.sql(
            "SELECT code_keys, env_lock FROM event_fit WHERE pipeline = ? "
            "AND label = ? ORDER BY seq DESC LIMIT 1",
            (self.pipeline.id, label),
        )
        if not rows:
            return "no earlier fit of this pipeline and label"
        before, now = json.loads(rows[0]["code_keys"]), self.stage_keys["fit"]
        files = sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))
        moved = [f"module changed: {', '.join(files)}"] if files else []
        if json.loads(rows[0]["env_lock"])["sha"] != self.env_lock["sha"]:
            moved.append("environment lock changed")
        return "; ".join(moved) or "train segments, members or feature columns changed"

    def _probe(
        self,
        stage: str,
        output: np.ndarray,
        compute: Callable[[Dataset, Range], Any],
        visible: Dataset,
        rows: Range,
    ):
        """Rerun the function on the rows before each cut of ``_cuts``, with its array
        inputs cut there, and require the predictions before it to stand; every computed
        range, before it is written.
        """
        lo, hi = rows
        for at in _cuts(visible, lo, hi):
            again = np.asarray(compute(visible.upto(at), (lo, at)))
            _refuse_changed(
                f"{self.name}: {stage}", "prediction", output, again, at=at, first=lo
            )

    def _one_per_row(self, stage: str, output: Any, rows: Range) -> np.ndarray:
        values = _one_per_row(f"{self.name}: {stage} returned", output, rows)
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
        _refuse_lazy_imports(self.pipeline, self.root, self.dists, loaded, stage)

    def _model(self, fit_id: str, fresh: bool = False) -> Any:
        """The fit's model loaded from its saved bytes once per run, or with ``fresh`` a
        new load, so a probe meets no state an earlier predict left in the model.
        """
        if fresh or fit_id not in self.models:
            fit = self.ledger.latest(Event.FIT, fit_id)
            model = self.pipeline.load(
                self.ledger.get_blob(fit["payload"]["model"]["sha"])
            )
            if fresh:
                return model
            self.models[fit_id] = model
        return self.models[fit_id]

    def _write(self, pred_id: str, pred: Any, payload: dict[str, Any]) -> np.ndarray:
        values = np.asarray(pred, dtype=np.float64)
        blob = {
            "sha": self.ledger.put_blob(formats.series_save(values)),
            "format": PARQUET,
        }
        self.ledger.append(
            Event.PREDICTIONS,
            self.evaluation.dataset_id,
            pred_id,
            {**payload, "blob": blob},
            id=pred_id,
        )
        self.report.predictions_computed += 1
        return values


def _score(
    context: _Run, folds: list[Fold], pipeline: Pipeline, predictions: dict[str, str]
):
    ledger, evaluation, report = context.ledger, context.evaluation, context.report
    name = pipeline.name or pipeline.id
    scorer = evaluation.scorer
    scoring = _scoring(evaluation)
    scorer_shas = identity.import_shas(scoring, context.root)
    dists = identity.imported_dists(scoring, context.root)
    lock = _lock_text(dists)
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
            "scorer_keys": identity.code_keys(scoring, context.root),
            "env_lock": identity.bytes_hash(lock.encode()),
            "directions": evaluation.directions,
        }
    )
    if ledger.latest(Event.SCORE, score_id) is not None:
        report.scores_reused += 1
        return
    if evaluation.sealed:
        _refuse_rescoring(ledger, evaluation, pipeline)
    if context.dry:
        report.scores_recorded += 1
        context.log(f"would score {name}")
        return
    per_fold, parts = [], []
    loaded = set(sys.modules)
    for index, fold in enumerate(folds):
        pred = _load_predictions(ledger, predictions[str(index)])
        visible = context.dataset.upto(fold.test[1])
        series = _series(
            f"{name}: the scorer's series on fold {fold.label} has",
            scorer.series(pred, visible, fold.test, evaluation.scorer_params),
        )
        parts.append(series)
        metrics = _finite(name, evaluation.metrics(series.copy()), f"fold {fold.label}")
        per_fold.append({"fold": index, "label": fold.label, "metrics": metrics})
    whole = np.concatenate(parts)
    aggregate = _finite(name, evaluation.metrics(whole.copy()), "the pooled folds")
    _refuse_lazy_imports(pipeline, context.root, dists, loaded, "scorer")
    columns = np.asarray(whole, np.float64).reshape(len(whole), -1)
    names = scorer.columns or [str(i) for i in range(columns.shape[1])]
    if len(names) != columns.shape[1]:
        raise Refused(
            f"scorer declares columns {list(names)}; its series has {columns.shape[1]}"
        )
    stored = {
        "sha": ledger.put_blob(
            formats.arrays_save(dict(zip(names, columns.T, strict=True)))
        ),
        "format": formats.Format.ARROW_ARRAYS,
        "fold_rows": [len(part) for part in parts],
    }
    try:
        ledger.append(
            Event.SCORE,
            evaluation.id,
            score_id,
            {
                "pipeline": pipeline.id,
                "run": context.start(),
                "predictions": predictions,
                "folds": per_fold,
                "aggregate": aggregate,
                "directions": evaluation.directions,
                "series": stored,
                "scorer_shas": scorer_shas,
                "sealed": evaluation.sealed,
            },
            id=score_id,
        )
    except sqlite3.IntegrityError:
        _refuse_rescoring(ledger, evaluation, pipeline)
        raise
    report.scores_recorded += 1
    context.log(
        f"score {name} " + " ".join(f"{k}={v:.4g}" for k, v in aggregate.items())
    )


def _cuts(dataset: Dataset, lo: int, hi: int) -> list[int]:
    """Where a probe cuts rows ``lo`` to ``hi``: at the timestamp after the open of a
    date that opens in the range (after the range's first timestamp when none does), and
    at any timestamp inside the range, each drawn by a hash of the range. The first cut
    catches a date's opening rows reading later rows of their date, or on a daily
    calendar the next date; the second varies where in its date a cut falls from range
    to range, so no calendar lines every cut up with a date open or a fixed time of day.
    A range of one timestamp has no cut, since its function sees nothing after it.
    """
    ts = np.arange(hi) if dataset.ts is None else dataset.ts[:hi]
    stamps = lo + np.flatnonzero(np.r_[True, ts[lo + 1 : hi] != ts[lo : hi - 1]])
    if len(stamps) < 2:
        return []
    starts = dataset.date_starts(max(lo - 1, 0), hi)
    opens = starts[(starts >= lo) & (starts < stamps[-1])]
    afters = stamps[np.searchsorted(stamps, opens if opens.size else [lo], "right")]
    draw = int(identity.content_hash([lo, hi]), 16)
    inner = stamps[1:]
    return sorted(
        {int(afters[draw % len(afters)]), int(inner[(draw >> 32) % len(inner)])}
    )


def _one_per_row(who: str, output: Any, rows: Range) -> np.ndarray:
    """``output`` as an array with one row per row of ``rows``, else refused."""
    values = np.asarray(output)
    n = rows[1] - rows[0]
    if values.ndim not in (1, 2) or values.shape[0] != n:
        raise Refused(
            f"{who} shape {values.shape}; it must have one row per row of the range, "
            f"shape ({n},) or ({n}, k)"
        )
    return values


def _series(who: str, output: Any) -> np.ndarray:
    """``output`` as a series, ``(rows,)`` or ``(rows, c)`` with at least one row, else
    refused.
    """
    values = np.asarray(output)
    if values.ndim == 0:
        raise Refused(
            f"{who} shape (): the series function returned a scalar; return one row "
            "per scoring unit, a row or a day"
        )
    if values.ndim > 2 or not len(values):
        raise Refused(
            f"{who} shape {values.shape}; a series is (rows,) or (rows, c) with at "
            "least one row, one row per scoring unit, a row or a day"
        )
    return values


def _noted(note: str, function: Callable, *args: Any) -> Any:
    """``function(*args)``, with ``note`` added to what it raises."""
    try:
        return function(*args)
    except Exception as error:
        error.add_note(note)
        raise


def _finite(name: str, metrics: dict[str, Any], where: str) -> dict[str, Any]:
    bad = sorted(
        k
        for k, v in metrics.items()
        if not isinstance(v, (int, float, np.number)) or not np.isfinite(v)
    )
    if bad:
        raise Refused(
            f"{name}: metrics {bad} are not finite on {where}; a metric is a finite "
            "number, so return one for an empty or constant series too"
        )
    return metrics


def _refuse_changed(
    who: str, name: str, full: Any, head: Any, at: int | None = None, first: int = 0
):
    """Refuse when ``head``, a function's output recomputed on the rows before ``at``
    (or, without ``at``, by the model reloaded from its saved bytes), differs from
    ``full`` by more than rounding: ``1e-9`` of the largest finite value of ``full``,
    or 1024 ulps of a narrower float dtype. Honest BLAS blocking and memory layout
    measure under 3e-12 of scale, the smallest leak 1e-7, so the band separates them
    without a per-element ulp rule, which is unsound near zero. NaN positions must
    agree. The one-sentence message names the first row that moved, both values, how
    many rows moved and the largest move as a fraction of scale, so rounding and a
    leak read differently.
    """
    dtype = np.asarray(full).dtype
    rtol = 1e-9 if dtype.kind != "f" else max(1e-9, 1024 * np.finfo(dtype).eps)
    full = np.asarray(full, dtype=np.float64)
    finite = np.abs(full[np.isfinite(full)])
    scale = float(finite.max()) if finite.size else 0.0
    full = full[: len(head)]
    head = np.asarray(head, dtype=np.float64)
    close = np.isclose(full, head, rtol=0, atol=rtol * scale, equal_nan=True)
    moved = np.flatnonzero(~close.reshape(len(head), -1).all(axis=1))
    if not moved.size:
        return
    r = moved[0]
    gap = np.abs(full[moved] - head[moved])
    worst = float(np.max(gap[np.isfinite(gap)], initial=0.0))
    row, fix = first + r, ""
    if at is None:
        where = (
            f"{who}: the model reloaded from its saved bytes predicts {name} "
            f"{_g(head[r])} at row {row} where the fitted one predicts {_g(full[r])}"
        )
        fix = ", so save must keep what predict reads and predict must be deterministic"
    else:
        where = (
            f"{who} read past row {at}: {name} at row {row} is {_g(full[r])} on the "
            f"whole input and {_g(head[r])} on the rows before {at}"
        )
    raise Refused(
        f"{where}, and {moved.size} of {len(head)} rows moved by up to {worst:.6g}, "
        f"{worst / scale if scale else np.inf:.6g} of the largest |{name}| "
        f"{scale:.6g} where rounding is allowed up to {rtol:.6g}{fix}"
    )


def _g(value: Any) -> str:
    """A float, or a row of floats, to six significant digits."""
    text = " ".join(f"{v:.6g}" for v in np.ravel(value))
    return text if np.ndim(value) == 0 else f"[{text}]"


def _refuse_overlap(folds: list[Fold]):
    for fold in folds:
        lo, hi = fold.test
        for segment in fold.train:
            if segment[0] < hi and lo < segment[1]:
                raise Refused(
                    f"fold {fold.label} trains on segment {tuple(segment)}, which "
                    f"overlaps its test range {tuple(fold.test)}"
                )


def _refuse_unrevealed(dataset: Dataset, targets: dict[str, Lag], folds: list[Fold]):
    """Refuse the folds that train on a row before their test range whose label is
    not known at the range's start, naming the worst one, the lag, and the purge
    (train rows dropped before each test range) that clears every fold at once.
    """
    worst: tuple[int, int, str] = (0, 0, "")
    leaky = 0
    for fold in folds:
        train = row_mask(dataset.rows, fold.train)
        lo = fold.test[0]
        known = dataset.upto(lo + 1).masked(targets, lo)
        for name in targets:
            late = np.flatnonzero(
                train[:lo]
                & np.isnan(known.columns[name][:lo])
                & ~np.isnan(dataset.columns[name][:lo])
            )
            if not late.size:
                continue
            leaky += 1
            stamps = len(np.unique(dataset.ts[late]))
            where = (
                f"fold {fold.label} trains on rows {late[0]} to {late[-1]} whose "
                f"{name} labels (reveal lag {_lag_text(targets[name])}) are not "
                f"known when its test range starts at row {lo}"
            )
            if (stamps, late.size) > worst[:2]:
                worst = (stamps, late.size, where)
    if leaky:
        stamps, rows, where = worst
        raise Refused(
            f"{leaky} folds train on labels revealed after their test range starts; "
            f"the worst: {where}; purge {stamps} more timestamps ({rows} rows) from "
            "the end of each train range, by raising the split's embargo_rows or "
            "embargo_timestamps, and they all clear; a label known at a later date's "
            "open takes an integer reveal lag, record(reveal={name: dates}), which is "
            "exact across holidays"
        )


def _refuse_sealed(evaluation: Evaluation, folds: list[Fold], sealed: int | None):
    """A sealed tail is validated only by a sealed evaluation, a sealed evaluation
    needs one and validates nothing else, and a fold trains on a tail row only after
    an earlier fold of a sealed evaluation tested it.
    """
    name = evaluation.name or evaluation.id
    if evaluation.sealed and sealed is None:
        raise Refused(
            f"evaluation {name} is sealed and dataset {evaluation.dataset_id} has no "
            "sealed tail; record the dataset with sealed_from"
        )
    if sealed is None:
        return
    tested = np.zeros(max(s[1] for f in folds for s in (*f.train, f.test)), bool)
    for fold in folds:
        lo, hi = fold.test
        if evaluation.sealed and lo < sealed:
            raise Refused(
                f"evaluation {name} is sealed and fold {fold.label} validates rows "
                f"{lo} to {hi}, before row {sealed} where the dataset's sealed tail "
                f"starts; a sealed evaluation validates the tail only, so start its "
                f"folds at row {sealed} or later"
            )
        if not evaluation.sealed and hi > sealed:
            raise Refused(
                f"evaluation {name} is not sealed and fold {fold.label} validates rows "
                f"{lo} to {hi}, past row {sealed} where the dataset's sealed tail "
                f"starts; end its folds by row {sealed}, or declare "
                "Evaluation(sealed=True)"
            )
        for a, b in fold.train:
            if b > sealed and not tested[max(a, sealed) : b].all():
                raise Refused(
                    f"evaluation {name} fold {fold.label} trains on rows {a} to {b}, "
                    f"past row {sealed} where the dataset's sealed tail starts; end "
                    f"its train segments by row {sealed}, or by the end of what an "
                    "earlier fold of a sealed evaluation tested"
                )
        if evaluation.sealed:
            tested[lo:hi] = True


def _held_score(ledger: Ledger, evaluation: Evaluation, pipeline: Pipeline) -> str:
    held = ledger.sql(
        "SELECT id FROM event_score WHERE evaluation = ? AND pipeline = ?",
        (evaluation.id, pipeline.id),
    )
    return held[0]["id"] if held else ""


def _refuse_rescoring(ledger: Ledger, evaluation: Evaluation, pipeline: Pipeline):
    held = _held_score(ledger, evaluation, pipeline)
    if held:
        raise Refused(
            f"sealed evaluation {evaluation.name or evaluation.id} already scored "
            f"pipeline {pipeline.name or pipeline.id} as score {held}; a "
            "sealed evaluation scores each pipeline once"
        )


def _lag_text(lag: Lag) -> str:
    if isinstance(lag, np.timedelta64):
        return str(datetime.timedelta(seconds=float(lag / np.timedelta64(1, "s"))))
    return f"{lag} dates"


def _lag(stored: Any) -> Lag:
    """A recipe's reveal lag: seconds as a timedelta, ``{"dates": n}`` as an int."""
    if stored is None:
        return None
    if isinstance(stored, dict):
        return int(stored["dates"])
    return np.timedelta64(round(stored * 1e9), "ns")


def _columns_id(columns: dict[str, np.ndarray]) -> str:
    """The ordered columns' names, dtypes and bytes: what a fit consumes."""
    return identity.content_hash(
        [[n, str(v.dtype), identity.array_hash(v)] for n, v in columns.items()]
    )


def _load_predictions(ledger: Ledger, pred_id: str) -> np.ndarray:
    event = ledger.latest(Event.PREDICTIONS, pred_id)
    return formats.series_load(ledger.get_blob(event["payload"]["blob"]["sha"]))


# Declarations and provenance ==========================================================


def _declare_pipeline(ledger: Ledger, pipeline: Pipeline):
    """One event per (id, name), so a rename reaches the views."""
    ledger.append(
        Event.PIPELINE,
        pipeline.id,
        pipeline.id,
        {
            "name": pipeline.name,
            "declaration": identity.canonical(pipeline),
            "params": identity.canonical(pipeline.params),
        },
        id=identity.content_hash({"id": pipeline.id, "name": pipeline.name}),
    )


def _declare_evaluation(
    ledger: Ledger, evaluation: Evaluation, folds: list[Fold], dry: bool
):
    """One event per (id, name); a split whose folds moved under an unchanged
    declaration is refused, since scores under one id must share one schedule.
    """
    live = [
        {
            "index": i,
            "label": f.label,
            "train": [list(seg) for seg in f.train],
            "test": f.test,
        }
        for i, f in enumerate(folds)
    ]
    before = ledger.latest(Event.EVALUATION, evaluation.id)
    if before is not None and before["payload"]["folds"] != json.loads(
        json.dumps(live)
    ):
        split = type(evaluation.split)
        raise Refused(
            f"{split.__module__}:{split.__qualname__} yields other folds than "
            f"evaluation {evaluation.id} recorded; rename it or change a field so the "
            "evaluation id changes, and the old evaluation and its scores stay"
        )
    if dry:
        return
    ledger.append(
        Event.EVALUATION,
        evaluation.id,
        evaluation.id,
        {
            "name": evaluation.name,
            "dataset": evaluation.dataset_id,
            "declaration": identity.canonical(evaluation),
            "metrics": evaluation.directions,
            "folds": live,
        },
        id=identity.content_hash({"id": evaluation.id, "name": evaluation.name}),
    )


def _refuse_unseen_class(obj: Any, shas: dict[str, str], root: pathlib.Path, by: str):
    """A dataclass in a params object, at any depth, defined in a repo module its
    consumer does not import would let an edited default reuse a stale fit; refuse and
    say where to define it.
    """
    root = root.resolve()
    for held in _held(obj):
        if not dataclasses.is_dataclass(held) or isinstance(held, type):
            continue
        file = getattr(sys.modules.get(type(held).__module__), "__file__", None)
        if not file:
            continue
        file = pathlib.Path(file).resolve()
        rel = file.relative_to(root).as_posix() if root in file.parents else None
        if rel is not None and rel not in shas:
            raise Refused(
                f"{type(held).__name__} is defined in {rel}, which the {by} function "
                "does not import, so a default edited there would not refit; define "
                f"it beside the {by} function"
            )


def _held(config: Any) -> list[Any]:
    """The dataclass instances and callables a params object holds at any depth, in
    dataclass fields, dict values, lists and tuples; their modules join the closure of
    the function that reads it.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        values = [getattr(config, f.name) for f in dataclasses.fields(config)]
        return [config, *(h for v in values for h in _held(v))]
    if isinstance(config, dict):
        return [h for v in config.values() for h in _held(v)]
    if isinstance(config, (list, tuple)):
        return [h for v in config for h in _held(v)]
    return [config] if callable(config) else []


def _stages(pipeline: Pipeline) -> dict[str, tuple[Any, ...]]:
    """What each stage's memo hashes: its functions and what their params hold."""
    post = pipeline.postprocess
    return {
        "features": pipeline.feature_functions,
        "fit": (
            pipeline.fit,
            pipeline.save,
            pipeline.load,
            *_held(pipeline.params),
        ),
        "predict": (pipeline.predict,),
        "postprocess": (post, *_held(pipeline.postprocess_config)) if post else (),
    }


def _scoring(evaluation: Evaluation) -> tuple[Any, ...]:
    """What the score's memo hashes: the scorer's functions and what its params
    hold.
    """
    scorer = evaluation.scorer
    return (scorer.series, scorer.metrics, *_held(evaluation.scorer_params))


def _nested(pipeline: Pipeline) -> Iterator[Pipeline]:
    yield pipeline
    for member in pipeline.members:
        yield from _nested(member)


def _positional(func: Callable) -> int:
    """Positional parameters of a function."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(p.kind in kinds for p in inspect.signature(func).parameters.values())


UNTRACKED_DIFF_CAP = 1 << 20
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
_SLOTS = ("fit", "predict", "save", "load", "postprocess")


def _git(ledger: Ledger, root: pathlib.Path) -> dict[str, Any]:
    """The commit, whether the tree is dirty, and one diff of every change against
    HEAD, untracked files included through a temporary index so git runs once; an
    untracked file over a mebibyte is named, not diffed.
    """

    def git(*args: str, env: dict[str, str] | None = None) -> bytes | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, check=True, env=env
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    def text(out: bytes | None) -> str:
        return (out or b"").decode(errors="surrogateescape")

    git_index = git("rev-parse", "--git-path", "index")
    if git_index is None:
        return {"commit": None, "dirty": None, "diff": None}
    head = text(git("rev-parse", "--verify", "-q", "HEAD")).strip() or None
    dirty = bool(text(git("status", "--porcelain")).strip())
    diff = None
    if dirty:
        paths = text(git("ls-files", "-z", "--others", "--exclude-standard")).split(
            "\0"
        )
        big = {
            p for p in paths if p and (root / p).lstat().st_size > UNTRACKED_DIFF_CAP
        }
        small = [p for p in paths if p and p not in big]
        with tempfile.TemporaryDirectory() as tmp:
            index = pathlib.Path(tmp) / "index"
            current = root / text(git_index).strip()
            if current.is_file():
                shutil.copy(current, index)
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            for at in range(0, len(small), 1000):
                git("add", "-N", "--", *small[at : at + 1000], env=env)
            patch = (git("diff", head or EMPTY_TREE, env=env) or b"") + "".join(
                f"# untracked, over {UNTRACKED_DIFF_CAP >> 20} MiB, not diffed: {p}\n"
                for p in big
            ).encode(errors="surrogateescape")
        diff = {"sha": ledger.put_blob(patch), "format": formats.Format.DIFF}
    return {"commit": head, "dirty": dirty, "diff": diff}


def _refuse_lazy_imports(
    pipeline: Pipeline,
    root: pathlib.Path,
    dists: dict[str, str],
    loaded: set,
    stage: str,
):
    """A function that imports inside its body hides code from the memo; refuse and
    name the repo modules, the modules outside the root no distribution owns, and the
    distributions it loaded.
    """
    new = set(sys.modules) - loaded
    root = root.resolve()
    grown = sorted(
        n
        for n in new
        if identity._module_path(sys.modules.get(n), root)
        or identity.unseen(sys.modules.get(n), root)
    )
    owners = identity.distribution_owners()
    tops = {name.partition(".")[0] for name in new}
    lazy = sorted({d for top in tops for d in owners.get(top, ()) if d not in dists})
    if grown or lazy:
        raise Refused(
            f"{pipeline.name or pipeline.id}: {stage} imported lazily: modules "
            f"{grown}, distributions {lazy}; import at module level"
        )


def _lock_text(dists: dict[str, str]) -> str:
    lines = [
        f"python=={platform.python_version()}",
        *(f"{n}=={v}" for n, v in dists.items()),
    ]
    return "\n".join(lines) + "\n"


def _resolution(ledger: Ledger, root: pathlib.Path) -> dict[str, str] | None:
    for path in [root / "uv.lock", *sorted(root.glob("requirements*.txt"))]:
        if path.exists():
            return {"path": path.name, **_text_blob(ledger, path.read_text())}
    return None


def _text_blob(ledger: Ledger, text: str) -> dict[str, str]:
    return {"sha": ledger.put_blob(text.encode()), "format": formats.Format.TEXT}


def _host() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "python": platform.python_version(),
    }
