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
    existed.
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
) -> RunReport:
    """Fit, predict and score each pipeline, reusing what the memo rule allows.

    ``run_started`` is appended only when something is computed, right before the first
    write, so a rerun of an unchanged tree writes nothing and adding one pipeline costs
    only its own fits, predictions and score. A pipeline that raises is recorded as
    ``pipeline_failed`` and the others continue; what it had written stays and a rerun
    resumes from there. Every step sees a prefix of the session: fit up to the end of
    its last train segment, predict, postprocess and the scorer up to the end of the
    window, so nothing after the cutoff can be read. Two names on one declaration are
    refused, since the ledger
    keeps one name per id and would score the second as a duplicate. ``log`` receives
    one line per fit, prediction set and score as it is written.
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
    session = dataset.load(ledger, evaluation.dataset)
    recipe = ledger.latest(Event.DATASET, evaluation.dataset)["payload"]["recipe"]
    targets = tuple(recipe["targets"])
    schedule = evaluation.split.folds(session)
    _declare_evaluation(ledger, evaluation, schedule)
    root = code_root or identity.repo_root(
        pathlib.Path(sys.modules[pipelines[0].fit.__module__].__file__)
    )
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

    for pipeline in pipelines:
        for declared in (pipeline, *pipeline.members):
            _declare_pipeline(ledger, declared)
        try:
            _run_pipeline(
                ledger,
                session,
                targets,
                schedule,
                pipeline,
                evaluation,
                root,
                report,
                start,
                log,
            )
        except Exception as error:  # noqa: BLE001
            name = pipeline.name or pipeline.id
            report.failed[name] = f"{type(error).__name__}: {error}"
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


def _run_pipeline(
    ledger: Ledger,
    session: Session,
    targets: tuple[str, ...],
    folds: list[Fold],
    pipeline: Pipeline,
    evaluation: Evaluation,
    root: pathlib.Path,
    report: RunReport,
    start: Callable[[], str],
    log: Callable[[str], None],
):
    stage = _Stage(
        ledger, session, targets, pipeline, evaluation, root, report, start, log
    )
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
    _score(
        ledger,
        session,
        folds,
        pipeline,
        evaluation,
        predictions,
        root,
        report,
        start,
        log,
    )


class _Stage:
    """One pipeline's memoized fits and predictions; a blend holds one per member."""

    def __init__(
        self,
        ledger: Ledger,
        session: Session,
        targets: tuple[str, ...],
        pipeline: Pipeline,
        evaluation: Evaluation,
        root: pathlib.Path,
        report: RunReport,
        start: Callable[[], str],
        log: Callable[[str], None],
    ):
        self.ledger, self.session, self.pipeline = ledger, session, pipeline
        self.targets, self.root = targets, root
        self.evaluation, self.report, self.start, self.log = (
            evaluation,
            report,
            start,
            log,
        )
        self.name = pipeline.name or pipeline.id
        self.shas = identity.import_shas(pipeline.steps, root)
        self.steps = {
            "fit": (pipeline.fit, pipeline.save, pipeline.load),
            "predict": (pipeline.predict,),
            "postprocess": (pipeline.postprocess,) if pipeline.postprocess else (),
        }
        self.stage_shas = {
            stage: identity.import_shas(steps, root)
            for stage, steps in self.steps.items()
        }
        self.takes_members = bool(pipeline.members) and _positional(pipeline.fit) > 3
        _refuse_unseen_class(pipeline.config, self.stage_shas["fit"], root, "fit")
        self.dists = identity.imported_dists(pipeline.steps, root)
        self.env_lock = _text_blob(ledger, _lock_text(self.dists))
        self.models: dict[str, Any] = {}
        self.fits: dict[str, str] = {}
        self.probed = False
        self.members = [
            _Stage(ledger, session, targets, m, evaluation, root, report, start, log)
            for m in pipeline.members
        ]

    def fit(self, fold: Fold) -> str:
        if fold.label in self.fits:
            return self.fits[fold.label]
        self.fits[fold.label] = fit_id = self._fit(fold)
        return fit_id

    def _fit(self, fold: Fold) -> str:
        train = [list(seg) for seg in fold.train]
        member_fits = [m.fit(fold) for m in self.members]
        fit_id = identity.content_hash(
            {
                "dataset": self.evaluation.dataset,
                "pipeline": self.pipeline.fit_declaration,
                "train": train,
                "import_shas": self.stage_shas["fit"],
                "env_lock": self.env_lock["sha"],
                **({"members": member_fits} if self.members else {}),
            }
        )
        if self.ledger.latest(Event.FIT, fit_id) is not None:
            self.report.fits_reused += 1
            return fit_id
        inputs = self._member_inputs(member_fits, fold) if self.takes_members else []
        run_id = self.start()
        started = time.perf_counter()
        visible = self.session.upto(max(hi for _, hi in fold.train))
        model = self.pipeline.fit(visible, fold.train, self.pipeline.config, *inputs)
        duration = time.perf_counter() - started
        self._unchanged("fit")
        self.models[fit_id] = model
        self.ledger.append(
            Event.FIT,
            self.evaluation.dataset,
            fit_id,
            {
                "pipeline": self.pipeline.id,
                "run": run_id,
                "train": train,
                "label": fold.label,
                "import_shas": self.shas,
                "env_lock": self.env_lock,
                "duration_s": duration,
                "model": {
                    "sha": self.ledger.put_blob(self.pipeline.save(model)),
                    "format": self.pipeline.format,
                    "portable": formats.KNOWN[self.pipeline.format],
                },
                **({"members": member_fits} if self.members else {}),
            },
            id=fit_id,
        )
        self.report.fits_computed += 1
        self.log(f"fit {self.name} {fold.label} {duration:.1f}s")
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
                "import_shas": self.stage_shas["predict"],
                **({"members": member_preds} if self.members else {}),
            }
        )
        pred_id = raw_id
        if self.pipeline.postprocess is not None:
            pred_id = identity.content_hash(
                {
                    "raw": raw_id,
                    "postprocess": self.pipeline.postprocess,
                    "import_shas": self.stage_shas["postprocess"],
                }
            )
        if self.ledger.latest(Event.PREDICTIONS, pred_id) is not None:
            self.report.predictions_reused += 1
            return pred_id
        self.start()
        where: dict[str, Any] = {
            "fit": fit_id,
            "range": list(rng),
            "fold": index,
            "window": window,
            **({"members": member_preds} if self.members else {}),
        }
        visible = self.session.upto(rng[1]).masked(self.targets, rng[0])
        if self.ledger.latest(Event.PREDICTIONS, raw_id) is not None:
            raw = _load_predictions(self.ledger, raw_id)
        else:
            inputs = [_load_predictions(self.ledger, p) for p in member_preds]
            extra = [inputs] if self.members else []
            model = self._model(fit_id)
            self._unchanged("predict")
            raw = self._write(
                raw_id, self.pipeline.predict(model, visible, rng, *extra), where
            )
            self.log(f"predictions {self.name} {window} rows {rng[0]}:{rng[1]}")
            self._probe(
                "predict",
                raw,
                lambda view: self.pipeline.predict(model, view, rng, *extra),
                visible,
                rng,
            )
        if self.pipeline.postprocess is not None:
            self._unchanged("postprocess")
            post = self.pipeline.postprocess(raw, visible, rng)
            self._probe(
                "postprocess",
                post,
                lambda view: self.pipeline.postprocess(raw, view, rng),
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

    def _member_inputs(
        self, member_fits: list[str], fold: Fold
    ) -> list[list[np.ndarray]]:
        """Each member's in-sample predictions over the train segments, one array
        each.
        """
        arrays = []
        for member, fit_id in zip(self.members, member_fits, strict=True):
            parts = [
                _load_predictions(
                    self.ledger,
                    member.predictions(fit_id, fold, -1, f"train:{k}", seg),
                )
                for k, seg in enumerate(fold.train)
            ]
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

    def _probe(
        self,
        stage: str,
        output: np.ndarray,
        compute: Callable[[Session], Any],
        visible: Session,
        rng: Range,
    ):
        """Once per pipeline, on its first computed window: freeze the rows after the
        window's midpoint and require the predictions before it to stand.
        """
        if self.probed or rng[1] - rng[0] < 2:
            return
        self.probed = True
        mid = (rng[0] + rng[1]) // 2
        again = np.asarray(compute(visible.frozen(mid)), dtype=np.float64)
        head = mid - rng[0]
        if not np.array_equal(output[:head], again[:head], equal_nan=True):
            raise Refused(
                f"{self.name}: {stage} reads rows after the one it predicts: "
                f"predictions before row {mid} changed when rows from {mid} were frozen"
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
    ledger: Ledger,
    session: Session,
    folds: list[Fold],
    pipeline: Pipeline,
    evaluation: Evaluation,
    predictions: dict[str, str],
    root: pathlib.Path,
    report: RunReport,
    start: Callable[[], str],
    log: Callable[[str], None],
):
    name = pipeline.name or pipeline.id
    scorer_shas = identity.import_shas((evaluation.scorer,), root)
    _refuse_unseen_class(evaluation.config, scorer_shas, root, "scorer")
    score_id = identity.content_hash(
        {
            "evaluation": evaluation.id,
            "pipeline": pipeline.id,
            "predictions": sorted(predictions.values()),
            "scorer_shas": scorer_shas,
        }
    )
    if ledger.latest(Event.SCORE, score_id) is not None:
        report.scores_reused += 1
        return
    per_fold, series = [], {}
    aggregate: dict[str, dict[str, float]] = {}
    for index, fold in enumerate(folds):
        for window, rng in fold.windows.items():
            pred = _load_predictions(ledger, predictions[f"{index}:{window}"])
            rows = np.asarray(
                evaluation.scorer(pred, session.upto(rng[1]), rng, evaluation.config)
            )
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
        stored[window] = {
            "sha": ledger.put_blob(
                formats.arrays_save({str(i): c for i, c in enumerate(columns.T)})
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
            "run": start(),
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
        log(
            f"score {name} {window} "
            + " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
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
    if not file or root.resolve() not in pathlib.Path(file).resolve().parents:
        return
    rel = pathlib.Path(file).resolve().relative_to(root.resolve()).as_posix()
    if rel not in shas:
        raise Refused(
            f"{type(obj).__name__} is defined in {rel}, which the {by} step does not "
            "import, so a default edited there would not refit; define it beside the "
            f"{by} step"
        )


def _positional(func: Callable) -> int:
    """Positional parameters of a step; kwargs bound by ``configured`` do not count."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    params = inspect.signature(func).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return 4
    return sum(p.kind in kinds for p in params)


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
