"""Runs: split a dataset, memoize fits and predictions, record a score per pipeline.

A run owns ranges, the clock and the scorer. A pipeline only sees ``fit``, ``predict``,
``save`` and ``load``. Reading is SQL over the views in ``ml_lab.ledger``.

Examples
--------
>>> report = run(ledger, [pipeline], evaluation)  # doctest: +SKIP
>>> report.fits_computed, report.scores_recorded  # doctest: +SKIP
(9, 1)
"""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import json
import os
import pathlib
import platform
import shutil
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
from ml_lab.dataset import Dataset, Lag, Range, load, row_mask, rows_save
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.splits import Fold


@dataclasses.dataclass
class RunReport:
    """What a run wrote; ``run`` is empty when every fit, prediction and score already
    existed. In a dry run the computed counts are what a run would compute.
    """

    run: str = ""
    fits_computed: int = 0
    predictions_computed: int = 0
    scores_recorded: int = 0
    fits_reused: int = 0
    predictions_reused: int = 0
    scores_reused: int = 0
    failed: dict[str, str] = dataclasses.field(default_factory=dict)


PARQUET = formats.Format.PARQUET


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
    only its own fits, predictions and score. A pipeline that raises is recorded as
    ``pipeline_failed`` and the others continue; what it had written stays and a rerun
    resumes from there. Every function sees a prefix of the dataset: fit up to the end
    of its last train segment, predict, postprocess and the scorer up to the end of the
    window, so nothing after the cutoff can be read. Two names on one declaration are
    refused, since the ledger keeps one name per id and would score the second as a
    duplicate. ``log`` receives one line per feature set, fit, prediction set and score
    as it is written. A ``dry`` run computes every id, looks each one up and writes
    nothing; it computes features it would need without the five-prefix probe, so
    every id is one a real run would write, and logs what it would compute and, for a
    fit, which repo files moved against the newest earlier fit of the same pipeline
    and label. ``planned`` carries the ids a dry run would write across several calls,
    so shared work counts once. A declaration error (a function that is not
    module-level, a slot holding something that is not a function, a config class its
    function does not import, a blend whose ``fit`` arity disagrees with
    ``in_sample``, ``in_sample`` without members), at any depth of blends inside
    blends, is refused before anything is read or written.
    """
    if not pipelines:
        raise Refused("no pipelines declared")
    names: dict[str, str] = {}
    for pipeline in pipelines:
        name = pipeline.name or pipeline.id
        if names.setdefault(pipeline.id, name) != name:
            raise Refused(
                f"pipelines {names[pipeline.id]} and {name} declare the same functions "
                f"and config, one id {pipeline.id}; rename or change one"
            )
        for p in _nested(pipeline):
            if p.in_sample and not p.members:
                raise Refused(f"pipeline {p.name or name}: in_sample needs members")
            for slot, held in [(s, getattr(p, s)) for s in _SLOTS] + [
                ("features", f) for f in p.feature_functions
            ]:
                if held is not None and not callable(held):
                    raise Refused(
                        f"pipeline {p.name or name}: {slot} holds a "
                        f"{type(held).__name__}, not a function"
                    )
            if p.members and (_positional(p.fit) > 3) != p.in_sample:
                raise Refused(
                    f"pipeline {p.name or name}: fit takes {_positional(p.fit)} "
                    f"positional parameters and in_sample={p.in_sample}; a blend fit "
                    "reads its members' train-range predictions as a fourth parameter "
                    "iff in_sample=True"
                )
        if pipeline.format not in formats.KNOWN:
            raise Refused(
                f"pipeline {name}: format {pipeline.format!r}, not one of "
                f"{sorted(formats.KNOWN)}"
            )
    root = code_root or identity.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
    for pipeline in pipelines:
        for p in _nested(pipeline):
            shas = identity.import_shas((p.fit, p.save, p.load), root)
            _refuse_unseen_class(p.config, shas, root, "fit")
            if p.postprocess is not None:
                shas = identity.import_shas((p.postprocess,), root)
                _refuse_unseen_class(p.postprocess_config, shas, root, "postprocess")
    scoring = (evaluation.scorer.score, evaluation.scorer.metrics)
    scorer_shas = identity.import_shas(scoring, root)
    _refuse_unseen_class(evaluation.config, scorer_shas, root, "scorer")
    dataset = load(ledger, evaluation.dataset)
    recipe = ledger.latest(Event.DATASET, evaluation.dataset)["payload"]["recipe"]
    targets = {name: _lag(lag) for name, lag in recipe["targets"].items()}
    schedule = evaluation.split.folds(dataset)
    _refuse_overlap(schedule)
    _refuse_unrevealed(dataset, targets, schedule)
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
            _run_pipeline(context, schedule, pipeline)
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = f"{type(error).__name__}: {error}"
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
        fit_id = stage.fit(fold)
        for window, rng in fold.windows.items():
            predictions[f"{index}:{window}"] = stage.predictions(
                fit_id, fold, index, window, rng
            )
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
            "features": self.feature_functions,
            "fit": (pipeline.fit, pipeline.save, pipeline.load),
            "predict": (pipeline.predict,),
            "postprocess": (pipeline.postprocess,) if pipeline.postprocess else (),
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
        self.lock_text = _lock_text(self.dists)
        self.env_lock = {
            "sha": identity.bytes_hash(self.lock_text.encode()),
            "format": formats.Format.TEXT,
        }
        self.feature_ids = [
            identity.content_hash(
                {
                    "dataset": self.evaluation.dataset,
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
        self.fits: dict[str, str] = {}
        self.checked: set[str] = set()
        self.widths: dict[str, tuple[int, ...]] = {}

    def fit(self, fold: Fold) -> str:
        if fold.label in self.fits:
            return self.fits[fold.label]
        self.fits[fold.label] = fit_id = self._fit(fold)
        return fit_id

    def _fit(self, fold: Fold) -> str:
        train = [list(seg) for seg in fold.train]
        member_fits = [m.fit(fold) for m in self.members]
        train_preds = (
            self._train_predictions(member_fits, fold)
            if self.pipeline.in_sample
            else []
        )
        columns_id = self._feature_columns_id()
        fit_id = identity.content_hash(
            {
                "dataset": self.evaluation.dataset,
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
        visible = self._visible(max(hi for _, hi in fold.train)).train_view(
            tuple(self.targets), fold.train
        )
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
        model = self.pipeline.fit(visible, fold.train, self.pipeline.config, *inputs)
        duration = time.perf_counter() - started
        blob = self._model_blob(model)
        self.models[fit_id] = self.pipeline.load(self.ledger.get_blob(blob["sha"]))
        self._check_imports("fit", loaded)
        self._unchanged("fit")
        self.ledger.put_blob(self.lock_text.encode())
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
        self, fit_id: str, fold: Fold, index: int, window: str, rng: Range
    ) -> str:
        member_fits = [m.fit(fold) for m in self.members]
        member_preds = [
            m.predictions(f, fold, index, window, rng)
            for m, f in zip(self.members, member_fits, strict=True)
        ]
        raw_id = identity.content_hash(
            {
                "fit": fit_id,
                "range": list(rng),
                "predict": self.pipeline.predict,
                "code_keys": self.stage_keys["predict"],
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
            "range": list(rng),
            "fold": index,
            "window": window,
            **({"members": member_preds} if self.members else {}),
        }
        visible = self._visible(rng[1]).masked(self.targets, rng[0])
        self.start()
        if self.ledger.latest(Event.PREDICTIONS, raw_id) is not None:
            raw = _load_predictions(self.ledger, raw_id)
        else:
            inputs = [_load_predictions(self.ledger, p) for p in member_preds]

            def predict(model: Any, view: Dataset, part: Range) -> np.ndarray:
                head = part[1] - rng[0]
                extra = [[a[:head] for a in inputs]] if self.members else []
                return np.asarray(self.pipeline.predict(model, view, part, *extra))

            loaded = set(sys.modules)
            model = self._model(fit_id)
            self._unchanged("predict")
            raw = self._one_per_row("predict", predict(model, visible, rng), rng)
            if fit_id in self.pending:
                self._record_fit(fit_id, fold, raw, lambda m: predict(m, visible, rng))
            self._probe(
                "predict", raw, lambda v, part: predict(model, v, part), visible, rng
            )
            self._check_imports("predict", loaded)
            raw = self._write(raw_id, raw, where)
            self.log(f"predictions {self.name} {window} rows {rng[0]}:{rng[1]}")
        if self.pipeline.postprocess is not None:
            postprocess = self.pipeline.postprocess
            config = self.pipeline.postprocess_config
            self._unchanged("postprocess")
            loaded = set(sys.modules)
            post = self._one_per_row(
                "postprocess", postprocess(raw, visible, rng, config), rng
            )
            self._probe(
                "postprocess",
                post,
                lambda view, part: postprocess(
                    raw[: part[1] - rng[0]], view, part, config
                ),
                visible,
                rng,
            )
            self._check_imports("postprocess", loaded)
            self._write(
                pred_id,
                post,
                {
                    **where,
                    "raw": raw_id,
                    "postprocess": identity.canonical(postprocess),
                    "postprocess_config": identity.canonical(config),
                },
            )
            self.log(f"postprocess {self.name} {window} rows {rng[0]}:{rng[1]}")
        return pred_id

    def _record_fit(
        self, fit_id: str, fold: Fold, loaded: np.ndarray, predict: Callable
    ):
        """Append a fit once its reloaded model predicts its first window as the
        fitted one did, so a save that drops state is refused, not recorded.
        """
        fitted, why, payload = self.pending.pop(fit_id)
        again = np.asarray(predict(fitted))
        _refuse_changed(
            f"{self.name}: fit {fold.label}: the model loaded from its saved bytes "
            "predicts differently from the fitted one (save must keep what predict "
            "reads, and predict must be deterministic)",
            "prediction",
            again,
            loaded,
        )
        self.ledger.append(
            Event.FIT, self.evaluation.dataset, fit_id, payload, id=fit_id
        )
        self.log(f"fit {self.name} {fold.label} {payload['duration_s']:.1f}s: {why}")

    def _train_predictions(self, member_fits: list[str], fold: Fold) -> list[list[str]]:
        return [
            [
                member.predictions(fit_id, fold, -1, f"train:{k}", seg)
                for k, seg in enumerate(fold.train)
            ]
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
        dataset without its targets, probed on five prefixes and stored; a dry run
        computes and stores nothing, so the names and id are known.
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
        probe_rows = sorted(
            {bare.boundary(j * n // 6) for j in range(1, 6)} - {0, None}
        )
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
            self.evaluation.dataset,
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
            "SELECT code_keys, env_lock FROM raw_fit WHERE pipeline = ? AND label = ? "
            "ORDER BY seq DESC LIMIT 1",
            (self.pipeline.id, label),
        )
        moved = ["no earlier fit of this pipeline and label"]
        if rows:
            before, now = json.loads(rows[0]["code_keys"]), self.stage_keys["fit"]
            moved = sorted(
                k for k in set(before) | set(now) if before.get(k) != now.get(k)
            )
            if json.loads(rows[0]["env_lock"])["sha"] != self.env_lock["sha"]:
                moved.append("environment lock")
        return ", ".join(moved) or "train segments, members or feature columns changed"

    def _probe(
        self,
        stage: str,
        output: np.ndarray,
        compute: Callable[[Dataset, Range], Any],
        visible: Dataset,
        rng: Range,
    ):
        """Rerun the function on the rows before the first row of the window's middle
        timestamp, with its array inputs cut there, and require the predictions
        before it to stand; every computed window, before it is written.
        """
        at = visible.boundary((rng[0] + rng[1]) // 2, rng[0])
        if at is None:
            return
        again = np.asarray(compute(visible.upto(at), (rng[0], at)))
        _refuse_changed(
            f"{self.name}: {stage}", "prediction", output, again, at=at, first=rng[0]
        )

    def _one_per_row(self, stage: str, output: Any, rng: Range) -> np.ndarray:
        values = np.asarray(output)
        rows = rng[1] - rng[0]
        if values.ndim not in (1, 2) or values.shape[0] != rows:
            raise Refused(
                f"{self.name}: {stage} returned shape {values.shape}; it must return "
                f"one row per row of the window, shape ({rows},) or ({rows}, k)"
            )
        width = self.widths.setdefault(stage, values.shape[1:])
        if values.shape[1:] != width:
            raise Refused(
                f"{self.name}: {stage} returned shape {values.shape} at rows "
                f"{rng[0]}:{rng[1]} and shape ({rows}, {', '.join(map(str, width))}) "
                f"on an earlier window; the width is a property of {stage}"
            )
        return values

    def _check_imports(self, stage: str, loaded: set[str]):
        """Check each stage once, on its first computation, before its write."""
        if stage not in self.checked:
            self.checked.add(stage)
            _refuse_lazy_imports(self.pipeline, self.root, self.dists, loaded, stage)

    def _model(self, fit_id: str) -> Any:
        if fit_id not in self.models:
            fit = self.ledger.latest(Event.FIT, fit_id)
            self.models[fit_id] = self.pipeline.load(
                self.ledger.get_blob(fit["payload"]["model"]["sha"])
            )
        return self.models[fit_id]

    def _write(self, pred_id: str, pred: Any, payload: dict[str, Any]) -> np.ndarray:
        values = np.asarray(pred, dtype=np.float64)
        blob = {
            "sha": self.ledger.put_blob(formats.series_save(values)),
            "format": PARQUET,
        }
        self.ledger.append(
            Event.PREDICTIONS,
            self.evaluation.dataset,
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
    scoring = (scorer.score, scorer.metrics)
    scorer_shas = identity.import_shas(scoring, context.root)
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
            "scorer_keys": identity.code_keys(scoring, context.root),
        }
    )
    if ledger.latest(Event.SCORE, score_id) is not None:
        report.scores_reused += 1
        return
    if context.dry:
        report.scores_recorded += 1
        context.log(f"would score {name}")
        return
    per_fold, series = [], {}
    aggregate: dict[str, dict[str, float]] = {}
    for index, fold in enumerate(folds):
        for window, rng in fold.windows.items():
            pred = _load_predictions(ledger, predictions[f"{index}:{window}"])
            visible = context.dataset.upto(rng[1])
            rows = np.asarray(scorer.score(pred, visible, rng, evaluation.config))
            series.setdefault(window, []).append(rows)
            per_fold.append(
                {
                    "fold": index,
                    "label": fold.label,
                    "window": window,
                    "metrics": evaluation.metrics(rows),
                }
            )
    stored: dict[str, dict[str, Any]] = {}
    for window, parts in series.items():
        whole = np.concatenate(parts)
        aggregate[window] = evaluation.metrics(whole)
        columns = np.asarray(whole, np.float64).reshape(len(whole), -1)
        names = scorer.columns or [str(i) for i in range(columns.shape[1])]
        if len(names) != columns.shape[1]:
            raise Refused(
                f"scorer declares columns {list(names)}; its series has "
                f"{columns.shape[1]}"
            )
        stored[window] = {
            "sha": ledger.put_blob(
                formats.arrays_save(dict(zip(names, columns.T, strict=True)))
            ),
            "format": formats.Format.ARROW_ARRAYS,
            "fold_rows": [len(part) for part in parts],
        }
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
        },
        id=score_id,
    )
    report.scores_recorded += 1
    for window, metrics in aggregate.items():
        context.log(
            f"score {name} {window} "
            + " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        )


def _refuse_changed(
    who: str, name: str, full: Any, head: Any, at: int | None = None, first: int = 0
):
    """Refuse when ``head``, a function's output recomputed on the rows before ``at``
    (or on another path to the same rows), differs from ``full`` by more than
    rounding: ``1e-9`` of the largest finite value of ``full``, or 1024 ulps of a
    narrower float dtype. Honest BLAS blocking and memory layout measure under 3e-12
    of scale, the smallest leak 1e-7, so the band separates them without a
    per-element ulp rule, which is unsound near zero. NaN positions must agree. The
    message names the first row that moved, both values, how many rows moved and the
    largest move as a fraction of scale, so rounding and a leak read differently in
    one line.
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
    if moved.size:
        r = moved[0]
        gap = np.abs(full[moved] - head[moved])
        worst = float(np.max(gap[np.isfinite(gap)], initial=0.0))
        where = f"read past row {at}" if at is not None else "is not stable"
        before = f"on the rows before {at}" if at is not None else "the second time"
        raise Refused(
            f"{who} {where}: {name} at row {first + r} is {full[r]!r} on the whole "
            f"input and {head[r]!r} {before}; {moved.size} of {len(head)} rows moved, "
            f"by up to {worst:.3g}, {worst / scale if scale else np.inf:.3g} of the "
            f"largest |{name}| {scale:.3g} (rounding is allowed up to {rtol:g})"
        )


def _refuse_overlap(folds: list[Fold]):
    for fold in folds:
        for segment in fold.train:
            for window, rng in fold.windows.items():
                if segment[0] < rng[1] and rng[0] < segment[1]:
                    raise Refused(
                        f"fold {fold.label} trains on segment {tuple(segment)}, which "
                        f"overlaps window {window} {tuple(rng)}"
                    )


def _refuse_unrevealed(dataset: Dataset, targets: dict[str, Lag], folds: list[Fold]):
    """Refuse the folds that train on a row before one of their windows whose label
    is not known at the window's start, naming the worst one, the lag, and the
    embargo that clears every fold at once.
    """
    worst: tuple[int, int, str] = (0, 0, "")
    leaky = 0
    for fold in folds:
        train = row_mask(dataset.rows, fold.train)
        for window, (lo, _) in fold.windows.items():
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
                    f"known when window {window} starts at row {lo}"
                )
                if (stamps, late.size) > worst[:2]:
                    worst = (stamps, late.size, where)
    if leaky:
        stamps, rows, where = worst
        raise Refused(
            f"{leaky} fold windows train on labels revealed after they start; the "
            f"worst: {where}; embargo {stamps} more timestamps ({rows} rows) than the "
            "split drops now and they all clear; a label known at a later date's open "
            "takes an integer reveal "
            "lag, record(reveal={name: dates}), which is exact across holidays"
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
            "config": identity.canonical(pipeline.config),
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
            "windows": f.windows,
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
            "dataset": evaluation.dataset,
            "declaration": identity.canonical(evaluation),
            "metrics": evaluation.directions,
            "folds": live,
        },
        id=identity.content_hash({"id": evaluation.id, "name": evaluation.name}),
    )


def _refuse_unseen_class(obj: Any, shas: dict[str, str], root: pathlib.Path, by: str):
    """A dataclass config defined in a repo module its consumer does not import would
    let an edited default reuse a stale fit; refuse and say where to define it.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return
    module = sys.modules.get(type(obj).__module__)
    file = getattr(module, "__file__", None)
    if not file:
        return
    file, root = pathlib.Path(file).resolve(), root.resolve()
    rel = file.relative_to(root).as_posix() if root in file.parents else None
    if rel is not None and rel not in shas:
        raise Refused(
            f"{type(obj).__name__} is defined in {rel}, which the {by} function does "
            "not import, so a default edited there would not refit; define it beside "
            f"the {by} function"
        )


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
    name the repo modules and distributions it loaded.
    """
    new = set(sys.modules) - loaded
    grown = sorted(
        n for n in new if identity._module_path(sys.modules.get(n), root.resolve())
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
