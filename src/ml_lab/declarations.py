"""Declarations as the log records them, and the shape of a pipeline's slots.

``declare_pipeline`` and ``declare_evaluation`` write one event per (id, name);
``stage_functions`` and ``scorer_functions`` say what each memo hashes; the refusals
here catch a declaration the memo could not protect.

Examples
--------
>>> stage_functions(pipeline)["fit"]  # doctest: +SKIP
(<function fit>, <function save>, <function load>)
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import pathlib
import sys
from collections.abc import Callable, Iterator
from typing import Any

from ml_lab import formats, identity
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.splits import Fold

# Public Functions ===================================================================


def declare_pipeline(ledger: Ledger, pipeline: Pipeline):
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


def declare_evaluation(
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


def project_root(pipelines: list[Pipeline]) -> pathlib.Path:
    """The git root of the first declared function outside this package.

    A blend the tool ships (``mean_blend``) listed first must not make the tool's
    own checkout the code root; the tool's root when every function is the tool's.
    """
    own = pathlib.Path(__file__).resolve().parent
    for pipeline in pipelines:
        for p in nested(pipeline):
            for function in p.functions:
                file = getattr(sys.modules[function.__module__], "__file__", None)
                if file and not pathlib.Path(file).resolve().is_relative_to(own):
                    return identity.repo_root(pathlib.Path(file))
    return identity.repo_root(own)


def refuse_half_declared(p: Pipeline, name: str):
    """A pipeline fits with save, load and a known format, or is a stateless blend:
    no fit, no save, load or format, and members.
    """
    label = p.name or name
    if p.fit is None:
        if not p.members:
            raise Refused(
                f"pipeline {label}: fit=None is a stateless blend, which needs members"
            )
        if p.save is not None or p.load is not None or p.format is not None:
            raise Refused(
                f"pipeline {label}: a stateless blend has no model, so no save, load "
                "or format"
            )
        if p.params is not None or p.prepare is not None:
            raise Refused(
                f"pipeline {label}: a stateless blend has no fit to read params or "
                "a prepare step; settings for its predict go in predict_config"
            )
        return
    if p.save is None or p.load is None:
        raise Refused(f"pipeline {label}: a fit needs save and load for its model")
    if p.format not in formats.KNOWN:
        raise Refused(
            f"pipeline {label}: format {p.format!r}, not one of {sorted(formats.KNOWN)}"
        )


def refuse_unseen_class(obj: Any, shas: dict[str, str], root: pathlib.Path, by: str):
    """A dataclass in a params object, at any depth, defined in a repo module its
    consumer does not import would let an edited default reuse a stale fit; refuse and
    say where to define it.
    """
    root = root.resolve()
    for held in holdings(obj):
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


def holdings(config: Any) -> list[Any]:
    """The dataclass instances and callables a params object holds at any depth, in
    dataclass fields, dict values, lists and tuples; their modules join the closure of
    the function that reads it.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        values = [getattr(config, f.name) for f in dataclasses.fields(config)]
        return [config, *(h for v in values for h in holdings(v))]
    if isinstance(config, dict):
        return [h for v in config.values() for h in holdings(v)]
    if isinstance(config, (list, tuple)):
        return [h for v in config for h in holdings(v)]
    return [config] if callable(config) else []


def stage_functions(pipeline: Pipeline) -> dict[str, tuple[Any, ...]]:
    """What each stage's memo hashes: its functions and what their params hold."""
    post = pipeline.postprocess
    return {
        "features": pipeline.feature_functions,
        "fit": tuple(
            f
            for f in (pipeline.prepare, pipeline.fit, pipeline.save, pipeline.load)
            if f is not None
        )
        + tuple(holdings(pipeline.params)),
        "predict": (pipeline.predict, *holdings(pipeline.predict_config)),
        "postprocess": (post, *holdings(pipeline.postprocess_config)) if post else (),
    }


def scorer_functions(evaluation: Evaluation) -> tuple[Any, ...]:
    """What the score's memo hashes: the scorer's functions and what its params
    hold.
    """
    scorer = evaluation.scorer
    return (scorer.series, scorer.metrics, *holdings(evaluation.scorer_params))


def nested(pipeline: Pipeline) -> Iterator[Pipeline]:
    yield pipeline
    for member in pipeline.members:
        yield from nested(member)


def positional(func: Callable) -> int:
    """Positional parameters of a function."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(p.kind in kinds for p in inspect.signature(func).parameters.values())


SLOTS = ("prepare", "fit", "predict", "save", "load", "postprocess")
