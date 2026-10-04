"""Typed declarations: steps, scorers, pipelines and evaluations.

Declarations are frozen dataclasses hashed by canonical serialization; see docs/spec.md. A
pipeline's name is a label and does not enter its hash. Identity is declaration only: code
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

from forestry import hashing


def step(func: Callable | None = None, *, format: str | None = None) -> Callable:
    """Register a fit, predict, filter, save or load function as a hashable step.

    Parameters
    ----------
    format : str, optional
        For a ``save`` step: the name of the byte format it writes, recorded on every model blob.
    """

    def decorate(f: Callable) -> Callable:
        return hashing.register(f, format=format)

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
        return hashing.register(func, metrics=metrics, directions=dict(directions))

    return decorate


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model, a model becomes predictions, and a model becomes bytes.

    ``save(model) -> bytes`` and ``load(bytes) -> model`` name a format that opens without Python.
    """

    fit: Callable
    predict: Callable
    save: Callable
    load: Callable
    config: Any
    name: str = dataclasses.field(default="", metadata={"label": True})

    @property
    def id(self) -> str:
        return hashing.content_hash(self)

    @property
    def format(self) -> str | None:
        return self.save.__forestry_meta__.get("format")

    @property
    def steps(self) -> tuple[Callable, ...]:
        return (self.fit, self.predict, self.save, self.load)

    def with_config(self, **changes: Any) -> Pipeline:
        """A copy with config fields replaced."""
        return dataclasses.replace(self, config=dataclasses.replace(self.config, **changes))

    def named(self, name: str) -> Pipeline:
        return dataclasses.replace(self, name=name)


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """How every pipeline on a dataset is scored: the walk-forward schedule and the scorer."""

    dataset: str
    scorer: Callable
    first_cutoff: str
    config: Any = None
    every_months: int = 1
    eval_months: int = 1
    ages: tuple[int, ...] = (1, 2, 3)
    embargo_seconds: int = 0
    min_folds: int = 3

    @property
    def id(self) -> str:
        return hashing.content_hash(self)

    @property
    def directions(self) -> dict[str, str]:
        return self.scorer.__forestry_meta__["directions"]

    @property
    def metrics(self) -> Callable:
        return self.scorer.__forestry_meta__["metrics"]
