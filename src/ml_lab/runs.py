"""Runs: split a session, memoize fits and predictions, record a score per pipeline.

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
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import dataset, formats, identity
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.session import Lag, Range, Session
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
    resumes from there. Every step sees a prefix of the session: fit up to the end of
    its last train segment, predict, postprocess and the scorer up to the end of the
    window, so nothing after the cutoff can be read. Two names on one declaration are
    refused, since the ledger keeps one name per id and would score the second as a
    duplicate. ``log`` receives one line per feature set, fit, prediction set and score
    as it is written. A ``dry`` run computes every id, looks each one up and writes
    nothing; it computes features it would need, so every id is one a real run
    would write, and logs what it would compute and, for a fit, which repo files moved
    against the newest earlier fit of the same pipeline and label. ``planned`` carries
    the ids a dry run would write across several calls, so shared work counts once.
    A declaration error (an unregistered step, a slot holding something that is not a
    step, a config class its step does not import, a blend whose ``fit`` arity
    disagrees with ``in_sample``) is refused before anything is read or written.
    """
    if not pipelines:
        raise Refused("no pipelines declared")
    names: dict[str, str] = {}
    for pipeline in pipelines:
        name = pipeline.name or pipeline.id
        if names.setdefault(pipeline.id, name) != name:
            raise Refused(
                f"pipelines {names[pipeline.id]} and {name} declare the same steps and "
                f"config, one id {pipeline.id}; rename or change one"
            )
        for p in (pipeline, *pipeline.members):
            held_by = [(s, getattr(p, s)) for s in ("fit", "predict", "save", "load")]
            held_by += [("postprocess", p.postprocess)]
            held_by += [("features", f) for f in p.feature_steps]
            for slot, held in held_by:
                if held is not None and not callable(held):
                    raise Refused(
                        f"pipeline {p.name or name}: {slot} holds a "
                        f"{type(held).__name__}, not a step"
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
                f"pipeline {name}: save step declares format {pipeline.format!r}, "
                f"not one of {sorted(formats.KNOWN)}"
            )
    root = code_root or identity.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
    for pipeline in pipelines:
        for p in (pipeline, *pipeline.members):
            shas = identity.import_shas((p.fit, p.save, p.load), root)
            _refuse_unseen_class(p.config, shas, root, "fit")
    scorer_shas = identity.import_shas((evaluation.scorer,), root)
    _refuse_unseen_class(evaluation.config, scorer_shas, root, "scorer")
    session = dataset.load(ledger, evaluation.dataset)
    recipe = ledger.latest(Event.DATASET, evaluation.dataset)["payload"]["recipe"]
    targets = {name: _lag(lag) for name, lag in recipe["targets"].items()}
    schedule = evaluation.split.folds(session)
    if not dry:
        _declare_evaluation(ledger, evaluation, schedule)
    report = RunReport()

    def start() -> str:
        if not report.run:
            report.run = identity.ulid()
            ledger.append(
                Event.RUN,
                evaluation.id,
                report.run,
                {
                    "git": _git(ledger, root),
                    "resolution": _resolution(ledger, root),
                    "host": _host(),
                    "pipelines": [{"id": p.id, "name": p.name} for p in pipelines],
                },
                id=report.run,
            )
        return report.run

    context = _Run(
        ledger,
        session,
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
            for declared in (pipeline, *pipeline.members):
                _declare_pipeline(ledger, declared)
        try:
            _run_pipeline(context, schedule, pipeline)
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = f"{type(error).__name__}: {error}"
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
    session: Session
    targets: dict[str, Lag]
    evaluation: Evaluation
    root: pathlib.Path
    report: RunReport
    start: Callable[[], str]
    log: Callable[[str], None]
    dry: bool
    planned: set[str]
    features: dict[str, Features] = dataclasses.field(default_factory=dict)


Features = tuple[tuple[str, ...], pathlib.Path | None, str]
"""One computed feature set: its column names, the Parquet file they are read from
(None in a dry run, which stores nothing) and the id of the ordered columns' bytes."""


def _run_pipeline(context: _Run, folds: list[Fold], pipeline: Pipeline):
    stage = _Stage(context, pipeline)
    report, root = context.report, context.root
    loaded: set[str] | None = set(sys.modules)
    predictions: dict[str, str] = {}
    for index, fold in enumerate(folds):
        computed_before = report.fits_computed
        fit_id = stage.fit(fold)
        if report.fits_computed > computed_before and loaded is not None:
            _refuse_lazy_imports(pipeline, root, stage.shas, stage.dists, loaded)
            loaded = None
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
        self.members = [_Stage(context, m) for m in pipeline.members]
        self.feature_steps = pipeline.feature_steps
        shared = {m.feature_steps for m in self.members}
        if not self.feature_steps and len(shared) == 1:
            self.feature_steps = shared.pop()
        self.shas = identity.import_shas(pipeline.steps, self.root)
        self.steps = {
            "features": self.feature_steps,
            "fit": (pipeline.fit, pipeline.save, pipeline.load),
            "predict": (pipeline.predict,),
            "postprocess": (pipeline.postprocess,) if pipeline.postprocess else (),
        }
        self.stage_shas = {
            stage: identity.import_shas(steps, self.root)
            for stage, steps in self.steps.items()
        }
        self.stage_keys = {
            stage: identity.code_keys(steps, self.root)
            for stage, steps in self.steps.items()
        }
        self.dists = identity.imported_dists(pipeline.steps, self.root)
        self.lock_text = _lock_text(self.dists)
        self.env_lock = {
            "sha": identity.bytes_hash(self.lock_text.encode()),
            "format": formats.Format.TEXT,
        }
        self.feature_ids = [
            identity.content_hash(
                {
                    "dataset": self.evaluation.dataset,
                    "features": step,
                    "code_keys": identity.code_keys((step,), self.root),
                    "env_lock": identity.bytes_hash(
                        _lock_text(identity.imported_dists((step,), self.root)).encode()
                    ),
                }
            )
            for step in self.feature_steps
        ]
        self.featured: Session | None = None
        self.models: dict[str, Any] = {}
        self.pending: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.fits: dict[str, str] = {}

    def fit(self, fold: Fold) -> str:
        if fold.label in self.fits:
            return self.fits[fold.label]
        self.fits[fold.label] = fit_id = self._fit(fold)
        return fit_id

    def _fit(self, fold: Fold) -> str:
        train = [list(seg) for seg in fold.train]
        member_fits = [m.fit(fold) for m in self.members]
        columns_id = self._feature_columns_id()
        fit_id = identity.content_hash(
            {
                "dataset": self.evaluation.dataset,
                "pipeline": self.pipeline.fit_declaration,
                "train": train,
                "code_keys": self.stage_keys["fit"],
                "env_lock": self.env_lock["sha"],
                **({"members": member_fits} if self.members else {}),
                **({"features": columns_id} if columns_id else {}),
            }
        )
        if fit_id in self.planned:
            return fit_id
        if self.ledger.latest(Event.FIT, fit_id) is not None:
            self.report.fits_reused += 1
            self.planned.add(fit_id)
            return fit_id
        inputs = (
            self._member_inputs(member_fits, fold) if self.pipeline.in_sample else []
        )
        if self.dry:
            self.planned.add(fit_id)
            self.report.fits_computed += 1
            self.log(f"would fit {self.name} {fold.label}: {self._why(fold.label)}")
            return fit_id
        visible = self._visible(max(hi for _, hi in fold.train))
        run_id = self.start()
        started = time.perf_counter()
        model = self.pipeline.fit(visible, fold.train, self.pipeline.config, *inputs)
        duration = time.perf_counter() - started
        self._unchanged("fit")
        blob = self._model_blob(model)
        self.models[fit_id] = self.pipeline.load(self.ledger.get_blob(blob["sha"]))
        self.ledger.put_blob(self.lock_text.encode())
        self.pending[fit_id] = (
            model,
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

            def predict(model: Any, view: Session, part: Range) -> np.ndarray:
                head = part[1] - rng[0]
                extra = [[a[:head] for a in inputs]] if self.members else []
                return np.asarray(self.pipeline.predict(model, view, part, *extra))

            model = self._model(fit_id)
            self._unchanged("predict")
            raw = predict(model, visible, rng)
            if fit_id in self.pending:
                self._record_fit(fit_id, fold, raw, lambda m: predict(m, visible, rng))
            self._probe(
                "predict", raw, lambda v, part: predict(model, v, part), visible, rng
            )
            raw = self._write(raw_id, raw, where)
            self.log(f"predictions {self.name} {window} rows {rng[0]}:{rng[1]}")
        if self.pipeline.postprocess is not None:
            self._unchanged("postprocess")
            post = np.asarray(self.pipeline.postprocess(raw, visible, rng))
            self._probe(
                "postprocess",
                post,
                lambda view, part: self.pipeline.postprocess(
                    raw[: part[1] - rng[0]], view, part
                ),
                visible,
                rng,
            )
            self._write(
                pred_id,
                post,
                {
                    **where,
                    "raw": raw_id,
                    "postprocess": identity.canonical(self.pipeline.postprocess),
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
        fitted, payload = self.pending.pop(fit_id)
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
        self.log(f"fit {self.name} {fold.label} {payload['duration_s']:.1f}s")

    def _member_inputs(
        self, member_fits: list[str], fold: Fold
    ) -> list[list[np.ndarray]]:
        """Each member's in-sample predictions over the train segments, one array
        each.
        """
        arrays = []
        for member, fit_id in zip(self.members, member_fits, strict=True):
            ids = [
                member.predictions(fit_id, fold, -1, f"train:{k}", seg)
                for k, seg in enumerate(fold.train)
            ]
            if self.dry:
                continue
            parts = [_load_predictions(self.ledger, p) for p in ids]
            arrays.append(np.concatenate(parts))
        return [arrays]

    def _unchanged(self, stage: str):
        """Refuse to record a sha for code that is not the code that ran."""
        now = identity.import_shas(self.steps[stage], self.root)
        moved = sorted(
            k
            for k in set(now) | set(self.stage_shas[stage])
            if now.get(k) != self.stage_shas[stage].get(k)
        )
        if moved:
            raise Refused(f"{self.name}: source changed during the run: {moved}; rerun")

    def _visible(self, cut: int) -> Session:
        """The session's first ``cut`` rows, with the pipeline's feature columns read
        on demand from their stored files.
        """
        if not self.feature_steps:
            return self.session.upto(cut)
        if self.featured is None:
            sets = [self._features(step, fid) for step, fid in self._feature_pairs()]
            names = tuple(n for columns, _, _ in sets for n in columns)
            stores = {n: path for columns, path, _ in sets for n in columns}
            self.featured = Session(
                self.session.columns, self.session.ts, names, stores
            )
        return self.featured.upto(cut)

    def _feature_pairs(self) -> list[tuple[Callable, str]]:
        return list(zip(self.feature_steps, self.feature_ids, strict=True))

    def _feature_columns_id(self) -> str | None:
        """The fit's feature input: the id of the ordered columns' bytes, computing the
        features first when no run has, so a dry run's ids are the real ones.
        """
        ids = [self._features(step, fid)[2] for step, fid in self._feature_pairs()]
        if len(ids) > 1:
            return identity.content_hash(ids)
        return ids[0] if ids else None

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

    def _features(self, step: Callable, feature_id: str) -> Features:
        """One feature step's columns, computed once per feature id over the session
        without its targets, probed on five prefixes and stored; a dry run computes
        and stores nothing, so the names and id are known.
        """
        if feature_id in self.features:
            return self.features[feature_id]
        event = self.ledger.latest(Event.FEATURES, feature_id)
        if event is not None:
            payload = event["payload"]
            path = self.ledger.blobs / payload["blob"]["sha"]
            found = (tuple(payload["columns"]), path, payload["columns_id"])
            self.features[feature_id] = found
            return found
        revealed = {k: v for k, v in self.targets.items() if v is not None}
        bare = Session(
            {
                k: v
                for k, v in self.session.columns.items()
                if k in revealed or k not in self.targets
            },
            self.session.ts,
        )
        started = time.perf_counter()
        columns = self._feature_columns(step, bare)
        duration = time.perf_counter() - started
        self._unchanged("features")
        columns_id = _columns_id(columns)
        if self.dry:
            self.features[feature_id] = (tuple(columns), None, columns_id)
            return self.features[feature_id]
        n = bare.rows
        probe_rows = sorted(
            {bare.boundary(j * n // 6) for j in range(1, 6)} - {0, None}
        )
        for row in probe_rows:
            head = self._feature_columns(step, bare.upto(row).masked(revealed, row - 1))
            for name, values in columns.items():
                try:
                    _refuse_changed(
                        f"{self.name}: features", name, values, head[name], at=row
                    )
                except Refused as error:
                    plain = self._feature_columns(step, bare.upto(row))[name]
                    if np.array_equal(plain, values[:row], equal_nan=True):
                        raise Refused(
                            f"{self.name}: features read a revealed target before its "
                            f"reveal lag: column {name} changes before row {row} when "
                            f"labels not yet known at row {row - 1} are hidden"
                        ) from error
                    raise
            del head
        run_id = self.start()
        sha = self.ledger.put_blob(formats.session_save(Session(columns, None)))
        self.ledger.append(
            Event.FEATURES,
            self.evaluation.dataset,
            feature_id,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "features": identity.canonical(step),
                "import_shas": identity.import_shas((step,), self.root),
                "code_keys": identity.code_keys((step,), self.root),
                "columns": list(columns),
                "columns_id": columns_id,
                "probe_rows": probe_rows,
                "duration_s": duration,
                "blob": {"sha": sha, "format": PARQUET},
            },
            id=feature_id,
        )
        self.log(f"features {self.name} {len(columns)} columns {duration:.1f}s")
        self.features[feature_id] = (
            tuple(columns),
            self.ledger.blobs / sha,
            columns_id,
        )
        return self.features[feature_id]

    def _feature_columns(self, step: Callable, bare: Session) -> dict[str, np.ndarray]:
        out = step(bare)
        clash = sorted(set(out) & set(self.session.columns))
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
        compute: Callable[[Session, Range], Any],
        visible: Session,
        rng: Range,
    ):
        """Rerun the step on the rows before the first row of the window's middle
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
    scorer_shas = identity.import_shas((evaluation.scorer,), context.root)
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
            "scorer_keys": identity.code_keys((evaluation.scorer,), context.root),
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
            visible = context.session.upto(rng[1])
            rows = np.asarray(evaluation.scorer(pred, visible, rng, evaluation.config))
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
        names = evaluation.scorer.__ml_lab_meta__.get("columns") or [
            str(i) for i in range(columns.shape[1])
        ]
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
    """Refuse when ``head``, a step's output recomputed on the rows before ``at`` (or
    on another path to the same rows), differs from ``full`` by more than rounding:
    ``1e-9`` of the largest finite value of ``full``, or 1024 ulps of a narrower float
    dtype. Honest BLAS blocking and memory layout measure under 3e-12 of scale, the
    smallest leak 1e-7, so the band separates them without a per-element ulp rule,
    which is unsound near zero. NaN positions must agree. The message names the first
    row that moved, both values, how many rows moved and the largest move as a
    fraction of scale, so rounding and a leak read differently in one line.
    """
    dtype = np.asarray(full).dtype
    rtol = 1e-9 if dtype.kind != "f" else max(1e-9, 1024 * np.finfo(dtype).eps)
    full = np.asarray(full, dtype=np.float64)
    finite = np.abs(full[np.isfinite(full)])
    scale = float(finite.max()) if finite.size else 0.0
    full = full[: len(head)]
    head = np.asarray(head, dtype=np.float64)
    close = np.isclose(full, head, rtol=0, atol=rtol * scale, equal_nan=True)
    moved = np.flatnonzero(~close)
    if moved.size:
        r = moved[0]
        gap = np.abs(full[moved] - head[moved])
        worst = float(np.nanmax(np.where(np.isfinite(gap), gap, np.nan), initial=0.0))
        where = f"read past row {at}" if at is not None else "is not stable"
        before = f"on the rows before {at}" if at is not None else "the second time"
        raise Refused(
            f"{who} {where}: {name} at row {first + r} is {full[r]!r} on the whole "
            f"input and {head[r]!r} {before}; {moved.size} of {len(head)} rows moved, "
            f"by up to {worst:.3g}, {worst / scale if scale else np.inf:.3g} of the "
            f"largest |{name}| {scale:.3g} (rounding is allowed up to {rtol:g})"
        )


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
        [
            [n, str(v.dtype), identity.bytes_hash(np.ascontiguousarray(v).tobytes())]
            for n, v in columns.items()
        ]
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


def _declare_evaluation(ledger: Ledger, evaluation: Evaluation, folds: list[Fold]):
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
        raise Refused(
            f"the split yields other folds than evaluation {evaluation.id} recorded; "
            "the split's code changed without its declaration, so rename it or delete "
            "the ledger"
        )
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
            f"{type(obj).__name__} is defined in {rel}, which the {by} step does not "
            "import, so a default edited there would not refit; define it beside the "
            f"{by} step"
        )


def _positional(func: Callable) -> int:
    """Positional parameters of a step; kwargs bound by ``configured`` do not count."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(p.kind in kinds for p in inspect.signature(func).parameters.values())


UNTRACKED_DIFF_CAP = 1 << 20


def _git(ledger: Ledger, root: pathlib.Path) -> dict[str, Any]:
    """The commit, whether the tree is dirty, and one diff of every change against
    HEAD, untracked files included through a temporary index so git runs once; an
    untracked file over a mebibyte is named, not diffed.
    """

    def git(*args: str, env: dict[str, str] | None = None) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "diff": None}
    dirty = bool((git("status", "--porcelain") or "").strip())
    diff = None
    if dirty:
        paths = (git("ls-files", "-z", "--others", "--exclude-standard") or "").split(
            "\0"
        )
        paths = [p for p in paths if p]
        big = [p for p in paths if (root / p).stat().st_size > UNTRACKED_DIFF_CAP]
        small = [p for p in paths if p not in set(big)]
        with tempfile.TemporaryDirectory() as tmp:
            index = pathlib.Path(tmp) / "index"
            current = root / (git("rev-parse", "--git-path", "index") or "").strip()
            if current.is_file():
                shutil.copy(current, index)
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            for at in range(0, len(small), 1000):
                git("add", "-N", "--", *small[at : at + 1000], env=env)
            text = (git("diff", "HEAD", env=env) or "") + "".join(
                f"# untracked, over {UNTRACKED_DIFF_CAP >> 20} MiB, not diffed: {p}\n"
                for p in big
            )
        diff = {"sha": ledger.put_blob(text.encode()), "format": formats.Format.DIFF}
    return {"commit": commit.strip(), "dirty": dirty, "diff": diff}


def _refuse_lazy_imports(
    pipeline: Pipeline,
    root: pathlib.Path,
    shas: dict[str, str],
    dists: dict[str, str],
    loaded: set,
):
    """A fit that imports inside the function hides code from the memo; refuse and name
    it.
    """
    grown = sorted(set(identity.import_shas(pipeline.steps, root)) - set(shas))
    owners = identity.distribution_owners()
    tops = {name.partition(".")[0] for name in set(sys.modules) - loaded}
    lazy = sorted({d for top in tops for d in owners.get(top, ()) if d not in dists})
    if grown or lazy:
        raise Refused(
            f"fit imported lazily: modules {grown}, distributions {lazy}; "
            "import at module level"
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
