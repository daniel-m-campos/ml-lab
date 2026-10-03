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

# Registration =====================================================================================


def step(func: Callable) -> Callable:
    """Register a fit, predict or filter function as a hashable step."""
    return hashing.register(func, kind="step")


def scorer(directions: Mapping[str, str], from_series: Callable | None = None) -> Callable:
    """Register a scorer: ``score(pred, session, range, config) -> ScoreResult``.

    Parameters
    ----------
    directions : Mapping[str, str]
        Metric name to ``"max"`` or ``"min"``; comparisons read it.
    from_series : Callable, optional
        ``from_series(series) -> dict`` recomputing metrics over the concatenated fold series;
        when absent the aggregate is the mean over folds.
    """

    def decorate(func: Callable) -> Callable:
        return hashing.register(
            func, kind="scorer", directions=dict(directions), from_series=from_series
        )

    return decorate


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    """One scorer call: the metric vector and an optional per-row series for aggregation."""

    metrics: dict[str, float]
    series: Any = None


# Pipeline =========================================================================================


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


# Evaluation =======================================================================================


@dataclasses.dataclass(frozen=True)
class Schedule:
    """Monthly cutoffs with eval windows at increasing ages and an embargo before each."""

    first_cutoff: str
    every_months: int = 1
    eval_months: int = 1
    ages: tuple[int, ...] = (1, 2, 3)
    embargo_seconds: int = 0


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """How every pipeline on a dataset is scored and compared."""

    dataset: str
    schedule: Schedule
    scorer: Callable
    config: Any = None
    min_folds: int = 3
    compare_age: int = 1

    @property
    def id(self) -> str:
        return hashing.content_hash(self)

    @property
    def directions(self) -> dict[str, str]:
        return self.scorer.__forestry_meta__["directions"]
