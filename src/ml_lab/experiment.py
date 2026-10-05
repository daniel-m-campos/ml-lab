"""What an experiment is made of: steps, scorers, pipelines and the evaluation.

An experiment module exposes ``pipelines`` and ``evaluations``; ``lab run`` reads both
by name. Declarations are frozen dataclasses hashed by canonical serialization; see
docs/spec.md. A name is a label and does not enter its hash; a field at its default is
left out, so adding a defaulted field keeps every id. Identity is declaration only: code
changes are caught by the fit memo, not by hashing files.

Examples
--------
>>> @step
... def fit(session, train, config): ...
>>> @step(format="arrow-arrays")
... def save(model) -> bytes: ...
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from ml_lab import identity


def step(func: Callable | None = None, *, format: str | None = None) -> Callable:
    """Register a fit, predict, filter, save or load function as a hashable step.

    Parameters
    ----------
    format : str, optional
        For a ``save`` step: the name of the byte format it writes, recorded on every
        model blob.
    """

    def decorate(f: Callable) -> Callable:
        return identity.register(f, format=format)

    return decorate(func) if func is not None else decorate


def scorer(
    metrics: Callable, directions: Mapping[str, str], columns: tuple[str, ...] = ()
) -> Callable:
    """Register ``score(pred, session, range, config) -> series``.

    The series is a 1-D or 2-D array with one row per scoring unit: a prediction, or a
    day for a cross-sectional metric. ``metrics`` defines the unit and must make sense
    over the folds' rows concatenated.

    Parameters
    ----------
    metrics : Callable
        ``metrics(series) -> dict``; applied per fold and over the concatenated folds.
    directions : Mapping[str, str]
        Metric name to ``"max"`` or ``"min"``; comparisons read it.
    columns : tuple[str, ...], optional
        Names for the series' columns, stored with the blob so it reads without the
        scorer's code; ``"0"``, ``"1"``, ... when absent. A count mismatch is refused.
    """

    def decorate(func: Callable) -> Callable:
        return identity.register(
            func, metrics=metrics, directions=dict(directions), columns=tuple(columns)
        )

    return decorate


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model, a model becomes predictions, and a model
    becomes bytes.

    The slots are the three scopes of a time-ordered evaluation, not a chain of
    transforms, and a step's slot says what it may read: ``features`` has dataset
    scope (the whole session without its target columns, computed and stored once per
    dataset, step, code and environment, however many pipelines and folds share it);
    ``fit`` has fold scope (the prefix up to its train end, once per train range);
    ``predict`` and ``postprocess`` have window scope (the prefix up to the window
    end with targets masked, once per fit and range). A step belongs in the broadest
    scope of what it reads, so a per-fold standardiser, or a lagged target whose
    reveal lag the dataset does not declare, lives in ``fit`` and ``predict``, and
    composition within a scope is the object ``fit`` returns, an sklearn pipeline
    included.

    ``save(model) -> bytes`` and ``load(bytes) -> model`` declare a format from
    ``formats.KNOWN``. ``features(session) -> {name: array}`` adds one column per
    array. ``postprocess(predictions, session, range)`` is the cheap stateless stage
    after ``predict``: neutralise, clip, rank. Its knobs are bound with
    ``step.configured(**kwargs)`` so they enter the prediction's identity and not the
    fit's. A pipeline with ``members`` is a blend: its ``fit`` and ``predict`` take a
    fourth argument, the members' predictions as a list of arrays: over the window for
    ``predict``, and over the train segments for ``fit`` only when ``fit`` declares a
    fourth positional parameter. Those are in-sample, so a weight learned on them
    overfits; a three-argument ``fit`` sets fixed weights and computes no train-range
    predictions. The members' fits and predictions are memoized on their own.

    Variants are ``dataclasses.replace``: ``replace(p, postprocess=clip.configured(
    at=3.0), name="gbt_clip")`` shares every fit and raw prediction with ``p``, since
    the fit id reads ``fit_declaration`` and the raw prediction id excludes
    ``postprocess``. The config class must live in a module the fit step imports, so
    an edited default is caught by the memo.
    """

    fit: Callable
    predict: Callable
    save: Callable
    load: Callable
    config: Any
    features: Callable | None = None
    postprocess: Callable | None = None
    members: tuple[Pipeline, ...] = ()
    name: str = dataclasses.field(default="", metadata={"label": True})

    @property
    def id(self) -> str:
        return identity.content_hash(self)

    @property
    def format(self) -> str | None:
        return self.save.__ml_lab_meta__.get("format")

    @property
    def steps(self) -> tuple[Callable, ...]:
        stages = (
            self.features,
            self.fit,
            self.predict,
            self.save,
            self.load,
            self.postprocess,
        )
        own = tuple(s for s in stages if s is not None)
        return own + tuple(s for m in self.members for s in m.steps)

    @property
    def fit_declaration(self) -> dict[str, Any]:
        """What a fit depends on: the fit, save and load steps and the config."""
        return {
            "fit": self.fit,
            "save": self.save,
            "load": self.load,
            "config": self.config,
        }

    def with_config(self, **changes: Any) -> Pipeline:
        """A copy with config fields replaced."""
        return dataclasses.replace(
            self, config=dataclasses.replace(self.config, **changes)
        )

    def named(self, name: str) -> Pipeline:
        return dataclasses.replace(self, name=name)


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """How every pipeline on a dataset is scored: a split into folds, a scorer and its
    config. Splits live in ``ml_lab.splits``; any frozen dataclass with
    ``folds(session) -> list[Fold]`` works. ``name`` is a label. A sweep is a list of
    ``dataclasses.replace(base, config=SimConfig(t), name=f"cost{t}")``; the scorer's
    config class must live in a module the scorer imports.
    """

    dataset: str
    split: Any
    scorer: Callable
    config: Any = None
    name: str = dataclasses.field(default="", metadata={"label": True})

    @property
    def id(self) -> str:
        return identity.content_hash(self)

    @property
    def directions(self) -> dict[str, str]:
        return self.scorer.__ml_lab_meta__["directions"]

    @property
    def metrics(self) -> Callable:
        return self.scorer.__ml_lab_meta__["metrics"]
