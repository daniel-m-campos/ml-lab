"""Typed declarations: steps, scorers, pipelines and evaluations.

Declarations are frozen dataclasses hashed by canonical serialization; see docs/spec.md. A
pipeline's name is a label and does not enter its hash.

Examples
--------
>>> @step
... def fit(session, train, config): ...
>>> Pipeline(name="p", fit=fit, predict=fit, config={"depth": 6}).id  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from forestry import hashing


def step(func: Callable) -> Callable:
    """Register a fit, predict or filter function as a hashable step."""
    return hashing.register(func, kind="step")


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
        return hashing.register(func, kind="scorer", metrics=metrics, directions=dict(directions))

    return decorate


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model and how a model becomes predictions."""

    fit: Callable
    predict: Callable
    config: Any
    name: str = dataclasses.field(default="", metadata={"label": True})

    @property
    def id(self) -> str:
        return hashing.content_hash(self)

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
