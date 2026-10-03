"""Typed declarations: steps, scorers, pipelines, evaluations, stages and gates.

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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from forestry import hashing

# Registration =====================================================================================


def step(func: Callable) -> Callable:
    """Register a fit, predict, filter or loss function as a hashable step."""
    return hashing.register(func, kind="step")


def scorer(directions: Mapping[str, str], from_series: Callable | None = None) -> Callable:
    """Register a scorer: ``score(pred, session, range, exec, config) -> ScoreResult``.

    Parameters
    ----------
    directions : Mapping[str, str]
        Metric name to ``"max"`` or ``"min"``; the Pareto rule reads it.
    from_series : Callable, optional
        ``from_series(series) -> dict`` recomputing metrics over concatenated fold series; when
        absent the aggregate is the mean over folds.
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
class Tune:
    """Search a space over an inner time-ordered split of the training range, then refit."""

    space: Mapping[str, Sequence[Any]]
    inner_months: int
    loss: Callable


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model and how a model becomes predictions."""

    fit: Callable
    predict: Callable
    config: Any
    name: str = dataclasses.field(default="", metadata={"label": True})
    heads: tuple[Any, ...] | None = None
    tune: Tune | None = None

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
class Sealed:
    """The final window, scored once at decision time."""

    months: int = 1
    pass_rule: str = "fold_range"
    fail: str = "no_deploy"


@dataclasses.dataclass(frozen=True)
class Gate:
    """What a candidate must do to leave a stage."""

    kind: str
    n: int | None = None
    metric: str | None = None
    minimum: float | None = None


class gates:
    """Gate constructors."""

    human = Gate("human")
    vs_baseline = Gate("vs_baseline")

    @staticmethod
    def pareto_front(n: int) -> Gate:
        return Gate("pareto_front", n=n)

    @staticmethod
    def top_k(n: int, metric: str) -> Gate:
        return Gate("top_k", n=n, metric=metric)

    @staticmethod
    def threshold(metric: str, minimum: float) -> Gate:
        return Gate("threshold", metric=metric, minimum=minimum)


@dataclasses.dataclass(frozen=True)
class Rule:
    """How two metric vectors are ordered: Pareto, or a priority order with a relative band."""

    kind: str = "pareto"
    order: tuple[str, ...] = ()
    band: float = 0.0


class rules:
    """Rule constructors."""

    pareto = Rule("pareto")

    @staticmethod
    def priority(*order: str, band: float = 0.02) -> Rule:
        """Decide on the first metric whose relative difference exceeds ``band``."""
        return Rule("priority", order=tuple(order), band=band)


@dataclasses.dataclass(frozen=True)
class Stage:
    """A scorer, its config, an optional execution grid and the gate out."""

    scorer: Callable
    gate: Gate
    config: Any = None
    grid: tuple[Any, ...] | None = None


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """How every candidate on a dataset is judged."""

    dataset: str
    schedule: Schedule
    stages: tuple[Stage, ...]
    min_folds: int = 3
    sealed: Sealed = Sealed()
    rule: Rule = Rule()
    compare_age: int = 1
    contracts: tuple[str, ...] = ("fit_predict",)

    @property
    def id(self) -> str:
        return hashing.content_hash(self)

    @property
    def final(self) -> Stage:
        return self.stages[-1]

    @property
    def directions(self) -> dict[str, str]:
        return self.final.scorer.__forestry_meta__["directions"]
