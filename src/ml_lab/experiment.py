"""What an experiment is made of: pipelines, the scorer and the evaluation.

An experiment module exposes ``pipelines`` and ``evaluations``; ``lab run`` reads both
by name. A pipeline's ``params``, an evaluation's ``scorer_params`` and a
``postprocess_config`` are frozen dataclasses when given. Declarations are frozen
dataclasses hashed by canonical serialization; see docs/spec.md. A function in a
declaration is any module-level function, named by its dotted path. A name is a label
and does not enter its hash; a field at its default is left out, so adding a defaulted
field keeps every id. Identity is declaration only: code changes are caught by the fit
memo, not by hashing files.

Examples
--------
>>> Pipeline(fit, predict, save, load, "arrow-arrays", params)  # doctest: +SKIP
>>> Scorer(squared_error, metrics=mean, directions={"mse": "min"})  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from ml_lab import identity
from ml_lab.identity import Refused


@dataclasses.dataclass(frozen=True)
class Scorer:
    """``series(pred, dataset, rows, scorer_params) -> series`` with the code that
    reads its series.

    The series is ``(rows,)`` or ``(rows, c)`` with one row per scoring unit: a
    prediction, or a day for a cross-sectional metric such as a daily Sharpe. A scalar
    or an empty series is refused. ``metrics`` runs per fold and over the concatenated
    folds, and the pooled value equals the row-weighted fold mean only for a metric
    that adds over rows. It reruns on the stored series alone, so the series carries
    what it needs. Only ``series`` enters the evaluation id; ``metrics`` and
    ``directions`` are code under the memo, and each score records the directions it
    was scored under. ``series`` is a module-level function; ``metrics`` may be a
    lambda. Keep the scorer in its own module: a fit's memo covers every module its
    functions import, so a scorer beside ``fit`` refits on a scorer edit.

    Parameters
    ----------
    series : Callable
        ``series(pred, dataset, rows, scorer_params) -> series``.
    metrics : Callable
        ``metrics(series) -> dict``; applied to a copy per fold and over the
        concatenated folds.
    directions : dict[str, str]
        Metric name to ``"max"`` or ``"min"``; comparisons read it.
    columns : tuple[str, ...], optional
        Names for the series' columns, stored with the blob so it reads without the
        scorer's code; ``"0"``, ``"1"``, ... when absent. A count mismatch is refused.
    """

    series: Callable
    metrics: Callable = dataclasses.field(metadata={"label": True})
    directions: dict[str, str] = dataclasses.field(metadata={"label": True})
    columns: tuple[str, ...] = dataclasses.field(default=(), metadata={"label": True})


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """How a training range becomes a model, a model becomes predictions, and a model
    becomes bytes.

    The slots are the three scopes of a time-ordered evaluation, not a chain of
    transforms, and a function's slot says what it may read. ``features`` has
    dataset scope: the whole dataset without its target columns, computed and stored
    once per dataset, function, code and environment. ``prepare`` and ``fit`` have
    fold scope: the prefix up to the train end with every column blank outside the
    train segments, once per train range. ``predict`` and ``postprocess`` have range
    scope: the prefix up to the end of the range they predict, with targets masked,
    once per fit and range. A function belongs in the broadest scope of what it
    reads, so a per-fold standardiser lives in ``fit`` and ``predict``, and
    composition within a scope is the object ``fit`` returns, an sklearn pipeline
    included.

    Variants are ``dataclasses.replace``: ``replace(p, postprocess=clip,
    postprocess_config=Clip(at=3.0), name="gbt_clip")`` shares every fit and raw
    prediction with ``p``, since the fit id reads ``fit_declaration`` and the raw
    prediction id excludes ``postprocess``. A params class must live in a module its
    function imports (``params`` beside ``fit``, ``predict_config`` beside
    ``predict``, ``postprocess_config`` beside ``postprocess``), so an edited default
    is caught by the memo.

    Parameters
    ----------
    fit : Callable | None
        ``fit(dataset, train, params) -> model``, with ``dataset`` a ``FitView``. A
        blend's ``fit`` may take a fourth positional parameter and then learns from
        its members: it receives their out-of-fold predictions over the train
        segments, ``splits.oof_folds`` cutting the train rows into contiguous blocks,
        each member fit on the rows outside a block, purged as the evaluation
        purges, and predicting it; every inner fit and prediction is memoized like
        the fold's own. ``fit=None`` declares a stateless blend: no
        model, no fit recorded, ``predict`` receives None as the model, and the
        predictions hash the members' fit ids; ``mean_blend`` declares one.
    predict : Callable
        ``predict(model, dataset, rows) -> predictions`` over the fold's test range.
        A blend's takes the members' predictions over the range as its fourth
        argument, and ``predict_config`` follows as the last argument when given.
    save, load : Callable | None
        ``save(model) -> bytes`` and ``load(bytes) -> model``, a round trip in
        ``format``; None only with ``fit=None``.
    format : str | None
        One of ``formats.KNOWN``; None only with ``fit=None``.
    params : Any
        A frozen dataclass handed to ``fit``; it enters the fit id.
    features : Callable | tuple[Callable, ...] | None
        ``features(dataset) -> {name: array}`` adds one column per array; a tuple
        adds each function's columns in order, every one memoized, probed and shared
        on its own, with no name collisions. The columns are read through
        ``dataset.column`` and ``dataset.matrix`` by the names in
        ``dataset.computed_columns``; a column no function reads is never loaded, so
        a learned selection lives in ``fit`` and costs only the columns it keeps.
        Their order is part of a fit's identity and of what the model sees (a
        booster samples and constrains features by position), so a reordering
        refits and may change results. A blend without its own ``features`` sees its
        members' computed columns when every member declares the same functions.
    postprocess : Callable | None
        ``postprocess(predictions, dataset, rows, postprocess_config)``, the cheap
        stateless stage after ``predict``: neutralise, clip, rank.
    postprocess_config : Any
        A frozen dataclass for ``postprocess``; it enters the prediction id and not
        the fit's.
    predict_config : Any
        A frozen dataclass for ``predict`` (a rule baseline's feature and sign, a
        seeded random baseline's seed); it enters the raw prediction id and not the
        fit's, so variants share fits.
    prepare : Callable | None
        ``prepare(dataset, train) -> object``, fit-time preprocessing pipelines
        share: a binned booster dataset, an encoded matrix, built once per train
        range, feature columns and prepare code within a run and handed to every
        ``fit`` that declares it as ``dataset.prepared``, so pipelines that differ in
        params pay for it once. It enters the fit id with ``fit``'s code and is not
        recorded: a run rebuilds it. Its arrays are frozen; a fit must not change it.
    members : tuple[Pipeline, ...]
        A blend's members, whose fits and predictions are memoized on their own.
    name : str
        A label; it does not enter the id.
    """

    fit: Callable | None
    predict: Callable
    save: Callable | None = None
    load: Callable | None = None
    format: str | None = None
    params: Any = None
    features: Callable | tuple[Callable, ...] | None = None
    postprocess: Callable | None = None
    postprocess_config: Any = None
    predict_config: Any = None
    prepare: Callable | None = None
    members: tuple[Pipeline, ...] = ()
    name: str = dataclasses.field(default="", metadata={"label": True})

    def __post_init__(self):
        if isinstance(self.features, tuple) and len(self.features) < 2:
            one = self.features[0] if self.features else None
            object.__setattr__(self, "features", one)

    @property
    def id(self) -> str:
        return identity.content_hash(self)

    @property
    def stateless(self) -> bool:
        """Whether the pipeline is a stateless blend: ``fit=None``, so no model."""
        return self.fit is None

    @property
    def learns_from_members(self) -> bool:
        """Whether ``fit`` receives the members' out-of-fold predictions: a blend whose
        ``fit`` takes a fourth positional parameter.
        """
        if self.fit is None or not self.members:
            return False
        return _positional(self.fit) > 3

    @property
    def feature_functions(self) -> tuple[Callable, ...]:
        """The features functions as a tuple, however declared."""
        if self.features is None:
            return ()
        return self.features if isinstance(self.features, tuple) else (self.features,)

    @property
    def functions(self) -> tuple[Callable, ...]:
        stages = (
            *self.feature_functions,
            self.prepare,
            self.fit,
            self.predict,
            self.save,
            self.load,
            self.postprocess,
        )
        own = tuple(s for s in stages if s is not None)
        return own + tuple(s for m in self.members for s in m.functions)

    @property
    def fit_declaration(self) -> dict[str, Any]:
        """What a fit depends on: the fit, save and load functions, the format and the
        params.
        """
        return {
            "fit": self.fit,
            "save": self.save,
            "load": self.load,
            "format": self.format,
            "params": self.params,
            **({"prepare": self.prepare} if self.prepare is not None else {}),
        }

    def with_params(self, **changes: Any) -> Pipeline:
        """A copy with params fields replaced."""
        return dataclasses.replace(
            self, params=dataclasses.replace(self.params, **changes)
        )

    def named(self, name: str) -> Pipeline:
        return dataclasses.replace(self, name=name)


def _positional(func: Callable) -> int:
    """Positional parameters of a function."""
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(p.kind in kinds for p in inspect.signature(func).parameters.values())


def mean_blend(*members: Pipeline, name: str = "") -> Pipeline:
    """A stateless blend: the mean of its members' predictions, no fit recorded.

    Parameters
    ----------
    *members : Pipeline
        At least two pipelines whose predictions are averaged.
    name : str, optional
        The blend's label; ``mean_blend(*seeds, name="gbt_seed5")`` is a seed set.

    Returns
    -------
    Pipeline
        The blend, with ``fit=None`` and ``mean_predict``.

    Raises
    ------
    Refused
        With fewer than two members.
    """
    if len(members) < 2:
        raise Refused("mean_blend takes at least two members")
    return Pipeline(fit=None, predict=mean_predict, members=members, name=name)


def mean_predict(
    model: None, dataset: Any, rows: tuple[int, int], members: list[Any]
) -> Any:
    """The members' predictions averaged, whatever their width.

    Parameters
    ----------
    model : None
        A stateless blend has no model.
    dataset : Any
        Unused; the view ``predict`` receives.
    rows : tuple[int, int]
        The range predicted.
    members : list
        The members' predictions over the range.

    Returns
    -------
    Any
        Their mean over the members' axis.
    """
    return np.mean(np.stack(members, axis=0), axis=0)


@dataclasses.dataclass(frozen=True)
class Fitted:
    """What ``fit`` returns when it has facts to record beside the model.

    The round count an inner validation chose, the best iteration, a learned
    threshold. The model is saved and loaded as if returned alone; the facts are
    recorded with the fit and shown as ``event_fit.facts``, so a per-fold choice
    reads with SQL instead of from the model bytes. Facts are provenance, not
    identity: they enter no id.

    Attributes
    ----------
    model : Any
        The model, as ``save`` takes it.
    facts : dict[str, Any]
        JSON values; a numpy array or scalar is recorded by its values.
    """

    model: Any
    facts: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """How every pipeline on a dataset is scored: a split into folds, a scorer and its
    params. ``dataset_id`` is a selector, resolved at run as ``lab run --dataset``
    resolves one: a source, an id or an id prefix. Splits live in ``ml_lab.splits``;
    any frozen dataclass with ``folds(dataset) -> list[Fold]`` works. ``name`` is a
    label. A sweep is a list of ``dataclasses.replace(base,
    scorer_params=SimParams(t), name=f"cost{t}")``; the scorer's params class must
    live in a module the scorer imports. Only a ``sealed``
    evaluation validates on the dataset's sealed tail (``record(sealed_from=...)``),
    it validates nothing before the tail, and it scores each pipeline once.
    """

    dataset_id: str
    split: Any
    scorer: Scorer
    scorer_params: Any = None
    sealed: bool = False
    name: str = dataclasses.field(default="", metadata={"label": True})

    @property
    def id(self) -> str:
        return identity.content_hash(self)
