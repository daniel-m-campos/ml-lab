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
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import dataset, formats, identity
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.session import Range, Session
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
    nothing; it logs what it would compute and, for a fit, which repo files moved
    against the newest earlier fit of the same pipeline and label. ``planned`` carries
    the ids a dry run would write across several calls, so shared work counts once.
    A declaration error (an unregistered step, a config class its step does not
    import) is refused before anything is read or written.
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
    targets = {
        name: None if lag is None else np.timedelta64(round(lag * 1e9), "ns")
        for name, lag in recipe["targets"].items()
    }
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
        session.boundary(min(f.train[-1][1] for f in schedule)) or 0,
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
    targets: dict[str, np.timedelta64 | None]
    evaluation: Evaluation
    root: pathlib.Path
    report: RunReport
    start: Callable[[], str]
    log: Callable[[str], None]
    probe_at: int
    dry: bool
    planned: set[str]


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

    def __init__(
        self,
        context: _Run,
        pipeline: Pipeline,
        loaded: dict[str, dict[str, np.ndarray]] | None = None,
    ):
        vars(self).update(vars(context))
        self.pipeline = pipeline
        self.loaded = {} if loaded is None else loaded
        self.name = pipeline.name or pipeline.id
        self.shas = identity.import_shas(pipeline.steps, self.root)
        self.steps = {
            "features": (pipeline.features,) if pipeline.features else (),
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
        self.takes_members = bool(pipeline.members) and _positional(pipeline.fit) > 3
        self.dists = identity.imported_dists(pipeline.steps, self.root)
        self.lock_text = _lock_text(self.dists)
        self.env_lock = {
            "sha": identity.bytes_hash(self.lock_text.encode()),
            "format": formats.Format.TEXT,
        }
        self.feature_id = None
        if pipeline.features is not None:
            lock = _lock_text(identity.imported_dists((pipeline.features,), self.root))
            self.feature_id = identity.content_hash(
                {
                    "dataset": self.evaluation.dataset,
                    "features": pipeline.features,
                    "code_keys": self.stage_keys["features"],
                    "env_lock": identity.bytes_hash(lock.encode()),
                }
            )
        self.featured: Session | None = None
        self.models: dict[str, Any] = {}
        self.pending: dict[str, tuple[Any, dict[str, Any]]] = {}
        self.fits: dict[str, str] = {}
        self.members = [_Stage(context, m, self.loaded) for m in pipeline.members]

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
                **({"features": columns_id} if self.feature_id else {}),
            }
        )
        if fit_id in self.planned or self.ledger.latest(Event.FIT, fit_id) is not None:
            self.report.fits_reused += 1
            return fit_id
        inputs = self._member_inputs(member_fits, fold) if self.takes_members else []
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
                **({"features": columns_id} if self.feature_id else {}),
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
        if pred_id in self.planned or self.ledger.latest(Event.PREDICTIONS, pred_id):
            self.report.predictions_reused += 1
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
                out = self.pipeline.predict(model, view, part, *extra)
                return np.asarray(out, dtype=np.float64)

            model = self._model(fit_id)
            self._unchanged("predict")
            raw = predict(model, visible, rng)
            if fit_id in self.pending:
                self._record_fit(fit_id, fold, raw, lambda m: predict(m, visible, rng))
            self._probe(
                "predict", raw, lambda v, part: predict(model, v, part), visible, rng
            )
            self._write(raw_id, raw, where)
            self.log(f"predictions {self.name} {window} rows {rng[0]}:{rng[1]}")
        if self.pipeline.postprocess is not None:
            self._unchanged("postprocess")
            post = np.asarray(
                self.pipeline.postprocess(raw, visible, rng), dtype=np.float64
            )
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
        again = np.asarray(predict(fitted), dtype=np.float64)
        if not np.array_equal(loaded, again, equal_nan=True):
            raise Refused(
                f"{self.name}: fit {fold.label}: the model loaded from its saved "
                "bytes predicts differently from the fitted one; save must keep "
                "what predict reads, and predict must be deterministic"
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
        """The session's first ``cut`` rows, with the pipeline's feature columns."""
        if self.pipeline.features is None:
            return self.session.upto(cut)
        if self.featured is None:
            added = self._features()
            self.featured = Session(
                {**self.session.columns, **added}, self.session.ts, tuple(added)
            )
        return self.featured.upto(cut)

    def _feature_columns_id(self) -> str | None:
        """The fit's feature input, the ordered columns' bytes; None in a dry run
        before the features exist, so the fit id is one that was never recorded.
        """
        if self.feature_id is None:
            return None
        event = self.ledger.latest(Event.FEATURES, self.feature_id)
        if event is None and not self.dry:
            self._visible(0)
            event = self.ledger.latest(Event.FEATURES, self.feature_id)
        return event and event["payload"]["columns_id"]

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

    def _features(self) -> dict[str, np.ndarray]:
        """The feature step's columns, computed once per feature id over the session
        without its targets and probed at the schedule's first cutoff.
        """
        if self.feature_id in self.loaded:
            return self.loaded[self.feature_id]
        event = self.ledger.latest(Event.FEATURES, self.feature_id)
        if event is not None:
            path = self.ledger.blobs / event["payload"]["blob"]["sha"]
            self.loaded[self.feature_id] = formats.session_load(path).columns
            return self.loaded[self.feature_id]
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
        columns = self._feature_columns(bare)
        duration = time.perf_counter() - started
        self._unchanged("features")
        before = bare.boundary(max(self.probe_at - 1, 0))
        probe_rows = sorted({before, self.probe_at} - {0, None})
        for row in probe_rows:
            head = self._feature_columns(bare.upto(row).masked(revealed, row - 1))
            for name, values in columns.items():
                _refuse_changed(f"{self.name}: features", name, row, values, head[name])
            del head
        run_id = self.start()
        self.ledger.append(
            Event.FEATURES,
            self.evaluation.dataset,
            self.feature_id,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "features": identity.canonical(self.pipeline.features),
                "import_shas": self.stage_shas["features"],
                "code_keys": self.stage_keys["features"],
                "columns": list(columns),
                "columns_id": _columns_id(columns),
                "probe_rows": probe_rows,
                "duration_s": duration,
                "blob": {
                    "sha": self.ledger.put_blob(
                        formats.session_save(Session(columns, None))
                    ),
                    "format": PARQUET,
                },
            },
            id=self.feature_id,
        )
        self.log(f"features {self.name} {len(columns)} columns {duration:.1f}s")
        self.loaded[self.feature_id] = columns
        return columns

    def _feature_columns(self, bare: Session) -> dict[str, np.ndarray]:
        out = self.pipeline.features(bare)
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
        if self.feature_id and not self.ledger.latest(Event.FEATURES, self.feature_id):
            moved.append("features not computed; reused if the columns are unchanged")
        return ", ".join(moved) or "train segments or members changed"

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
        again = np.asarray(compute(visible.upto(at), (rng[0], at)), dtype=np.float64)
        _refuse_changed(
            f"{self.name}: {stage}", "prediction", at, output, again, rng[0]
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
    who: str, name: str, row: int, full: Any, head: Any, first: int = 0
):
    """Refuse when a step's output before ``row`` moved once the rows from ``row``
    were gone; name the first row that moved and both values.
    """
    full = np.asarray(full, dtype=np.float64)[: len(head)]
    head = np.asarray(head, dtype=np.float64)
    moved = np.flatnonzero((full != head) & ~(np.isnan(full) & np.isnan(head)))
    if moved.size:
        r = moved[0]
        raise Refused(
            f"{who} read past row {row}: {name} at row {first + r} is {full[r]} on "
            f"the whole input and {head[r]} on the rows before {row}"
        )


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


def _git(ledger: Ledger, root: pathlib.Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "diff": None}
    dirty = bool((git("status", "--porcelain") or "").strip())
    diff = None
    if dirty:
        untracked = (git("ls-files", "--others", "--exclude-standard") or "").split()
        text = (git("diff", "HEAD") or "") + "".join(
            subprocess.run(
                ["git", "diff", "--no-index", "--", os.devnull, path],
                cwd=root,
                capture_output=True,
                text=True,
            ).stdout
            for path in untracked
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
