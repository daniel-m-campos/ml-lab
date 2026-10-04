"""What an experiment is made of: steps, scorers, pipelines and the evaluation.

An experiment module exposes ``pipelines`` and ``evaluation``; ``lab run`` reads both by
name. Declarations are frozen dataclasses hashed by canonical serialization; see
docs/spec.md. A pipeline's name is a label and does not enter its hash. Identity is
declaration only: code changes are caught by the fit memo, not by hashing files.

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


def scorer(metrics: Callable, directions: Mapping[str, str]) -> Callable:
    """Register ``score(pred, session, range, config) -> series``.

    Parameters
    ----------
    metrics : Callable
        ``metrics(series) -> dict``; applied per fold and over the concatenated folds.
    directions : Mapping[str, str]
        Metric name to ``"max"`` or ``"min"``; comparisons read it.
    """

    def decorate(func: Callable) -> Callable:
        return identity.register(func, metrics=metrics, directions=dict(directions))

    return decorate


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model, a model becomes predictions, and a model
    becomes bytes.

    ``save(model) -> bytes`` and ``load(bytes) -> model`` declare a format from
    ``formats.KNOWN``. ``postprocess(predictions, session, range)`` is the optional
    cheap stage after ``predict``: neutralise, clip, rank. Its knobs are bound with
    ``step.configured(**kwargs)`` so they enter the prediction's identity and not the
    fit's. A pipeline with ``members`` is a blend: its ``fit`` and ``predict`` take a
    fourth argument, the members' predictions as a list of arrays (over the train
    segments for ``fit``, over the window for ``predict``), and the members' fits and
    predictions are memoized on their own.
    """

    fit: Callable
    predict: Callable
    save: Callable
    load: Callable
    config: Any
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
        stages = (self.fit, self.predict, self.save, self.load, self.postprocess)
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
    ``folds(session) -> list[Fold]`` works.
    """

    dataset: str
    split: Any
    scorer: Callable
    config: Any = None

    @property
    def id(self) -> str:
        return identity.content_hash(self)

    @property
    def directions(self) -> dict[str, str]:
        return self.scorer.__ml_lab_meta__["directions"]

    @property
    def metrics(self) -> Callable:
        return self.scorer.__ml_lab_meta__["metrics"]
