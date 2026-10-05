"""The ledger story on synthetic data: ingest, declare, run, read back with SQL.

Every event type in docs/spec.md is written and read back; what each function refuses is
pinned here.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import importlib
import io
import json
import math
import os
import pathlib
import py_compile
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import types
from collections.abc import Iterator

import numpy as np
import polars as pl
import pytest

from ml_lab import cli, formats, identity, runs, splits
from ml_lab.dataset import (
    Dataset,
    Range,
    Segments,
    data_id,
    load,
    record,
    rows_load,
    rows_save,
)
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.panel import Panel
from ml_lab.runs import _load_predictions
from tests import synthetic

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ml-lab")


@pytest.fixture
def dataset(ledger: Ledger) -> str:
    return synthetic.dataset(ledger)


@pytest.fixture
def evaluation(dataset: str):
    return synthetic.evaluation(dataset)


@pytest.fixture
def scratch(request) -> Iterator[pathlib.Path]:
    """A directory under the repo, so inside a run's code root, that git ignores."""
    path = REPO / ".scratch" / re.sub(r"\W", "_", request.node.name)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


def _run(ledger, evaluation, *pipelines, **kwargs):
    return runs.run(ledger, list(pipelines), evaluation, code_root=REPO, **kwargs)


def _count(ledger, source: str, *params) -> int:
    return ledger.sql(f"SELECT COUNT(*) AS n FROM {source}", params)[0]["n"]


def _types(ledger) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in ledger.events():
        out[e["type"]] = out.get(e["type"], 0) + 1
    return out


def _load_series(ledger, sha: str) -> np.ndarray:
    return np.column_stack(list(formats.arrays_load(ledger.get_blob(sha)).values()))


def _latest(ledger, evaluation) -> dict[str, str]:
    rows = ledger.sql(
        "SELECT p.name, l.score FROM score_latest l "
        "JOIN raw_pipeline p ON p.id = l.pipeline "
        "WHERE l.evaluation = ?",
        (evaluation.id,),
    )
    return {r["name"]: r["score"] for r in rows}


def _repo(
    root: pathlib.Path, files: dict[str, str] | None = None, commit: bool = True
) -> pathlib.Path:
    for name, text in (files or {"a.py": "x = 1\n"}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git = ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    if commit:
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "a"], check=True)
    return root


def _editable(
    tmp_path: pathlib.Path,
    monkeypatch,
    name: str,
    single: bool = False,
    study: pathlib.Path | None = None,
) -> pathlib.Path:
    """An editable install ``name`` of a git repo, and a study module ``<name>_steps``
    in ``study`` whose fit reads ``a.SCALE``; returns the module or package the install
    serves.

    A package holds modules ``a`` and ``b``; a ``single`` module install is ``a``.
    """
    site, repo = tmp_path / "site", tmp_path / "tool"
    study = study or tmp_path / "study"
    info = site / f"{name}-0.1.dist-info"
    for directory in (info, study):
        directory.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n"
    )
    (info / "direct_url.json").write_text(
        json.dumps({"url": repo.as_uri(), "dir_info": {"editable": True}})
    )
    if single:
        _repo(repo, {f"{name}.py": "SCALE = 1.0\n"})
    else:
        _repo(
            repo,
            {
                f"{name}/__init__.py": "",
                f"{name}/a.py": "SCALE = 1.0\n",
                f"{name}/b.py": "def main():\n    return 0\n",
            },
        )
    imported = f"import {name} as a" if single else f"from {name} import a"
    (study / f"{name}_steps.py").write_text(
        f"{imported}\n"
        "from tests import synthetic\n"
        "def fit(dataset, train, config):\n"
        "    model = synthetic.ridge_fit(dataset, train, config)\n"
        "    return synthetic.RidgeModel(model.weights * a.SCALE, model.bias)\n"
    )
    for path in (site, repo, study):
        monkeypatch.syspath_prepend(path)
    return repo / (f"{name}.py" if single else name)


def _module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(textwrap.dedent(source), module.__dict__)
    return module


def _dataset(n: int = 48, **extra: np.ndarray) -> Dataset:
    rng = np.random.default_rng(0)
    columns = {"x": rng.standard_normal(n), "y": rng.standard_normal(n), **extra}
    clock = np.datetime64("2025-01-01", "ns") + np.arange(n).astype("timedelta64[h]")
    return Dataset(columns, clock)


def _record(ledger: Ledger, dataset: Dataset, **reveal) -> str:
    return record(
        ledger,
        dataset,
        source="s",
        params={},
        filters=(),
        targets=("y",),
        reveal=reveal or None,
    )


def _strings(*values: str | None) -> np.ndarray:
    return np.array([None if v is None else "".join(list(v)) for v in values], object)


def nan_safe_blend_fit(dataset: Dataset, train, config, members) -> object:
    x = np.nan_to_num(np.column_stack(members))
    y = dataset.column(synthetic.TARGET, train)
    return synthetic.BlendModel(np.linalg.lstsq(x, y, rcond=None)[0])


def _nested(member):
    inner = dataclasses.replace(
        synthetic.blend(member, synthetic.ridge(3)), fit=nan_safe_blend_fit, name="in"
    )
    return dataclasses.replace(
        synthetic.blend(inner, synthetic.ridge(6)), fit=nan_safe_blend_fit, name="out"
    )


LAZY_STEPS = """
from tests import synthetic


def lazy_fit(dataset, train, config):
    import {helper}
    return synthetic.ridge_fit(dataset, train, config)


def lazy_predict(model, dataset, rng):
    import {helper}
    return synthetic.ridge_predict(model, dataset, rng)
"""


def _lazy_steps(scratch, tag: str):
    helper = f"lazy_helper_{tag}"
    (scratch / f"{helper}.py").write_text("K = 1\n")
    code = scratch / f"lazy_steps_{tag}.py"
    code.write_text(LAZY_STEPS.format(helper=helper))
    return cli._load(str(code)), helper


def _score(ledger: Ledger, score: str, pipeline: str, values: list[float]):
    folds = [
        {"fold": i, "label": str(i), "metrics": {"m": v}} for i, v in enumerate(values)
    ]
    payload = {"pipeline": pipeline, "folds": folds, "aggregate": {"m": 0.0}}
    ledger.append(Event.SCORE, "e", score, payload, id=score)


# Dataset ==============================================================================


def test_ingesting_the_same_recipe_twice_appends_one_event(ledger):
    assert synthetic.dataset(ledger) == synthetic.dataset(ledger)
    assert _types(ledger) == {Event.DATASET: 1}


def test_the_dataset_id_is_the_data_not_the_recipe(ledger, dataset):
    same_rows = synthetic.dataset(ledger, seed=7)
    assert same_rows == dataset
    kwargs = dict(filters=(synthetic.keep_all,), targets=(synthetic.TARGET,))
    rows = synthetic.generate(
        start="2025-01-01", months=12, rows_per_day=20, seed=7, drift_at="2025-07-01"
    )
    assert record(ledger, rows, source="other", params={"v": 2}, **kwargs) == dataset
    assert synthetic.dataset(ledger, seed=8) != dataset
    assert _types(ledger) == {Event.DATASET: 2}


def test_the_dataset_blob_opens_with_polars_alone(ledger, dataset):
    event = ledger.latest(Event.DATASET, dataset)
    assert event["payload"]["blob"]["format"] == "parquet"
    table = pl.read_parquet(
        io.BytesIO(ledger.get_blob(event["payload"]["blob"]["sha"]))
    )
    assert (
        synthetic.TARGET in table.columns and table.height == event["payload"]["rows"]
    )
    assert ledger.sql("SELECT source FROM raw_dataset")[0] == {"source": "synthetic"}


def test_a_target_that_is_not_numeric_is_refused_at_ingest(ledger):
    rows = Dataset({"x": np.arange(4.0), "class": _strings("1", "2", "1", "2")})
    with pytest.raises(TypeError, match="encode labels as numbers"):
        record(ledger, rows, source="toy", params={}, filters=(), targets=("class",))
    flags = Dataset({"x": np.arange(4.0), "up": np.array([True, False, True, False])})
    assert record(ledger, flags, source="toy", params={}, filters=(), targets=("up",))


def test_a_string_column_hashes_by_its_values_not_its_objects():
    def dataset(*values):
        return Dataset({"c": _strings(*values), "x": np.arange(3.0)})

    a = dataset("ab", None, "c")
    assert data_id(a) == data_id(dataset("ab", None, "c"))
    assert data_id(a) != data_id(dataset("ab", "", "c"))
    assert data_id(a) != data_id(dataset("ab", None, "d"))
    assert data_id(rows_load(rows_save(a))) == data_id(a)
    with pytest.raises(Refused, match="not a str, number or None"):
        data_id(Dataset({"c": np.array([b"raw", "s"], object)}))


def test_a_column_named_ts_is_refused(ledger):
    dataset = Dataset({"ts": np.arange(5.0), "y": np.ones(5)})
    with pytest.raises(ValueError, match="clock's name"):
        rows_save(dataset)
    with pytest.raises(ValueError, match="clock's name"):
        _record(ledger, dataset)


def test_record_refuses_zero_rows(ledger):
    with pytest.raises(ValueError, match="at least one row"):
        _record(ledger, _dataset(0))


def test_a_reveal_lag_is_a_non_negative_timedelta_or_an_int_from_one(ledger):
    dataset_id = _record(ledger, _dataset(), y=np.timedelta64(3600 * 10**9, "ns"))
    payload = ledger.latest(Event.DATASET, dataset_id)["payload"]
    assert payload["recipe"]["targets"]["y"] == 3600.0
    with pytest.raises(TypeError, match="float"):
        _record(ledger, _dataset(), y=1.5)
    for lag in (datetime.timedelta(hours=-1), np.timedelta64(-1, "h")):
        with pytest.raises(ValueError, match="negative"):
            _record(ledger, _dataset(), y=lag)
    for lag in (0, -1):
        with pytest.raises(Refused, match="starts at 1; a same-date reveal is a"):
            _record(ledger, _dataset(), y=lag)


def test_record_refuses_no_targets_and_a_column_that_is_not_1d(ledger):
    with pytest.raises(Refused, match="at least one target; without one nothing"):
        record(ledger, _dataset(), source="s", params={}, filters=(), targets=())
    wide = _dataset(4, w=np.zeros((4, 2)))
    with pytest.raises(Refused, match=r"\{'w': \(4, 2\)\}; a column is 1-D"):
        _record(ledger, wide)


def test_a_sealed_tail_starts_at_a_row_kept_in_the_payload_and_the_id(ledger):
    open_id = _record(ledger, _dataset())
    by_row = record(
        ledger,
        _dataset(),
        source="s",
        params={},
        filters=(),
        targets=("y",),
        sealed_from=40,
    )
    by_date = record(
        ledger,
        _dataset(),
        source="s",
        params={},
        filters=(),
        targets=("y",),
        sealed_from="2025-01-02T15:30",
    )
    assert by_row == by_date != open_id
    assert ledger.latest(Event.DATASET, by_row)["payload"]["sealed_from"] == 40
    assert ledger.latest(Event.DATASET, open_id)["payload"]["sealed_from"] is None
    with pytest.raises(Refused, match="a sealed tail starts after the first row"):
        record(
            ledger,
            _dataset(),
            source="s",
            params={},
            filters=(),
            targets=("y",),
            sealed_from="2026-01-01",
        )


def test_an_integer_reveal_lag_counts_dates_with_rows(ledger):
    rows = synthetic.generate(
        start="2025-01-01", months=12, rows_per_day=20, seed=7, drift_at="2025-07-01"
    )
    kw = dict(source="synthetic", params={}, filters=(), targets=(synthetic.TARGET,))
    by_date = record(ledger, rows, reveal={synthetic.TARGET: 1}, **kw)
    by_time = record(
        ledger, rows, reveal={synthetic.TARGET: datetime.timedelta(1)}, **kw
    )
    assert by_date != by_time != record(ledger, rows, **kw)
    recipe = ledger.events(Event.DATASET, key=by_date)[0]["payload"]["recipe"]
    assert recipe["targets"][synthetic.TARGET] == {"dates": 1}
    evaluation = synthetic.evaluation(by_date)
    base = synthetic.ridge(1)
    known = synthetic.featured(base, synthetic.label_one_date_back, "known")
    early = synthetic.featured(base, synthetic.label_one_row_back, "early")
    report = _run(ledger, evaluation, known, early)
    assert set(report.failed) == {"early"}
    assert "before its reveal lag" in report.failed["early"]
    dataset = load(ledger, by_date)
    at = dataset.index_of("2025-03-03")
    masked = dataset.masked({synthetic.TARGET: 1}, at).columns[synthetic.TARGET]
    assert np.isnan(masked[at:]).all() and not np.isnan(masked[:at]).any()
    masked = dataset.masked({synthetic.TARGET: 2}, at).columns[synthetic.TARGET]
    assert np.isnan(masked[at - 20]) and not np.isnan(masked[at - 21])


# Declarations =========================================================================


def test_a_pipeline_name_is_a_label_and_a_config_change_is_a_new_id():
    a = synthetic.ridge(3)
    assert a.named("other").id == a.id
    assert a.with_config(alpha=2.0).id != a.id


def test_adding_a_defaulted_field_keeps_the_id_and_a_rename_reaches_the_views(
    ledger, evaluation
):
    old = dataclasses.make_dataclass("Config", [("alpha", float)], frozen=True)
    new = dataclasses.make_dataclass(
        "Config",
        [
            ("alpha", float),
            ("include", tuple, dataclasses.field(default=())),
            ("clip", float, dataclasses.field(default=math.nan)),
        ],
        frozen=True,
    )
    assert identity.content_hash(old(1.0)) == identity.content_hash(new(1.0))
    assert identity.content_hash(new(1.0, ("x",))) != identity.content_hash(old(1.0))
    _run(ledger, evaluation, synthetic.ridge(1))
    _run(ledger, evaluation, synthetic.ridge(1).named("renamed"))
    assert set(_latest(ledger, evaluation)) == {"renamed"}
    assert _count(ledger, "raw_pipeline") == 1
    assert len(ledger.events(Event.PIPELINE)) == 2


def test_a_config_class_its_function_does_not_import_is_refused(ledger, evaluation):
    @dataclasses.dataclass(frozen=True)
    class Far:
        train_window_months: int = 3
        alpha: float = 1.0

    far = dataclasses.replace(synthetic.ridge(3), config=Far())
    with pytest.raises(Refused, match="define it beside the fit function"):
        _run(ledger, evaluation, synthetic.ridge(1), far)
    deep = dataclasses.replace(
        synthetic.ridge(3), config=synthetic.RidgeConfig(3, 1.0, ({"far": Far()},))
    )
    with pytest.raises(Refused, match="Far is defined in tests/test_ledger_story.py"):
        _run(ledger, evaluation, deep)
    with pytest.raises(Refused, match="define it beside the scorer function"):
        _run(ledger, dataclasses.replace(evaluation, config=Far()), synthetic.ridge(1))
    scaled = dataclasses.replace(
        synthetic.ridge(3), postprocess=synthetic.scale, postprocess_config=Far()
    )
    with pytest.raises(Refused, match="define it beside the postprocess function"):
        _run(ledger, evaluation, scaled)
    assert _types(ledger) == {Event.DATASET: 1}


HELD_STEPS = """
import dataclasses

from tests import synthetic


@dataclasses.dataclass(frozen=True)
class Cfg:
    knobs: tuple = ()


def fit(dataset, train, config):
    model = synthetic.ridge_fit(dataset, train, synthetic.RidgeConfig(1, 1.0))
    return synthetic.RidgeModel(model.weights * config.knobs[0]["by"](), model.bias)
"""

HELD_KNOBS = """
import dataclasses


def scale():
    return 1.0


@dataclasses.dataclass(frozen=True)
class Inner:
    scale: float = 1.0
"""


def test_a_function_a_config_holds_at_any_depth_joins_the_fit_closure(
    ledger, evaluation, scratch
):
    (scratch / "steps_held.py").write_text(HELD_STEPS)
    (scratch / "knobs_held.py").write_text(HELD_KNOBS)
    steps = cli._load(str(scratch / "steps_held.py"))
    knobs = cli._load(str(scratch / "knobs_held.py"))
    config = steps.Cfg(({"by": knobs.scale},))
    held = dataclasses.replace(synthetic.ridge(1), fit=steps.fit, config=config)
    first = _run(ledger, evaluation, held)
    assert first.fits_computed > 0 and _run(ledger, evaluation, held).run == ""
    (scratch / "knobs_held.py").write_text(HELD_KNOBS.replace("1.0", "-1.0", 1))
    assert _run(ledger, evaluation, held).fits_computed == first.fits_computed
    inner = dataclasses.replace(held, config=steps.Cfg((knobs.Inner(),)))
    with pytest.raises(Refused, match="Inner is defined in .*knobs_held.py, which the"):
        _run(ledger, evaluation, inner)


@pytest.mark.parametrize(
    "broken",
    [
        dataclasses.replace(synthetic.ridge(1), features=("f0_lag",)),
        _nested(dataclasses.replace(synthetic.ridge(1), predict="ridge_predict")),
    ],
    ids=["features", "member of a member's predict"],
)
def test_a_slot_that_is_not_a_function_is_refused_before_anything_is_written(
    ledger, evaluation, broken
):
    with pytest.raises(Refused, match="holds a str, not a function"):
        _run(ledger, evaluation, broken)
    assert _types(ledger) == {Event.DATASET: 1}


def test_a_lambda_a_closure_or_a_partial_in_a_slot_is_refused(ledger, evaluation):
    def local_fit(dataset, train, config):
        return synthetic.ridge_fit(dataset, train, config)

    for slot, held in [
        ("features", lambda dataset: {}),
        ("fit", local_fit),
        ("predict", functools.partial(synthetic.ridge_predict)),
    ]:
        p = dataclasses.replace(synthetic.ridge(1), **{slot: held})
        with pytest.raises(Refused, match="is not a module-level function of"):
            _run(ledger, evaluation, p)
    message = "<lambda>' is not a module-level function of tests.test_ledger_story"
    with pytest.raises(Refused, match=message):
        identity.step_ref(lambda: None)
    assert identity.step_ref(synthetic.ridge_fit) == "tests.synthetic:ridge_fit"
    assert _types(ledger) == {Event.DATASET: 1}


@pytest.mark.parametrize("dry", [False, True])
def test_a_split_whose_folds_moved_under_one_declaration_is_refused(
    ledger, dataset, evaluation, monkeypatch, dry
):
    _run(ledger, evaluation, synthetic.ridge(1))
    folds = evaluation.split.folds(load(ledger, dataset))
    monkeypatch.setattr(type(evaluation.split), "folds", lambda self, s: folds[1:])
    with pytest.raises(Refused, match="yields other folds") as refused:
        _run(ledger, evaluation, synthetic.ridge(3), dry=dry)
    assert "ml_lab.splits:CalendarWalkForward" in str(refused.value)
    assert "delete" not in str(refused.value)


def test_a_blend_whose_fit_takes_three_arguments_predicts_no_train_range(
    ledger, evaluation
):
    fixed = dataclasses.replace(
        synthetic.blend(synthetic.ridge(1), synthetic.ridge(3)),
        fit=synthetic.equal_fit,
        in_sample=False,
        name="equal",
    )
    _run(ledger, evaluation, fixed)
    assert _count(ledger, "raw_prediction") == _count(ledger, "raw_fit")
    assert "equal" in _latest(ledger, evaluation)
    with pytest.raises(Refused, match="in_sample"):
        _run(ledger, evaluation, dataclasses.replace(fixed, in_sample=True))
    learned = dataclasses.replace(fixed, fit=synthetic.blend_fit, in_sample=False)
    with pytest.raises(Refused, match="fourth parameter"):
        _run(ledger, evaluation, learned)


def test_in_sample_without_members_is_refused_before_anything_is_written(
    ledger, evaluation
):
    p = dataclasses.replace(synthetic.ridge(1), in_sample=True)
    with pytest.raises(Refused, match="in_sample needs members"):
        _run(ledger, evaluation, p)
    assert _count(ledger, "event") == 1


def test_a_dry_run_counts_shared_work_once_as_the_run_does(
    ledger, dataset, scratch, capsys
):
    shared = scratch / "shared.py"
    shared.write_text(
        "import dataclasses\nfrom tests import synthetic\n"
        "one, three = synthetic.ridge(1), synthetic.ridge(3)\n"
        "scaled = dataclasses.replace(one, postprocess=synthetic.scale, "
        "postprocess_config=2.0, name='scaled')\n"
        "pipelines = [one, three, scaled, synthetic.blend(one, three)]\n"
        "def evaluations(d):\n"
        "    return [dataclasses.replace(synthetic.evaluation(d, cost=c), name=f'c{c}')"
        " for c in (0.001, 0.01)]\n"
    )
    args = ["--root", str(ledger.root), "run", str(shared)]
    counts = r"fits (\d+), predictions (\d+), scores (\d+)"
    assert cli.main([*args, "--dry-run"]) == 0
    dry = re.findall(counts, capsys.readouterr().out)
    assert cli.main(args) == 0
    real = re.findall(counts, capsys.readouterr().out)
    assert dry == real and dry[1] == ("0", "0", "4")


def test_lab_run_refuses_a_lambda_once(ledger, dataset, tmp_path, capsys):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import dataclasses\nfrom tests import synthetic\n"
        "bare = lambda dataset: {}\n"
        "pipelines = [dataclasses.replace(synthetic.ridge(3), features=bare)]\n"
        "def evaluations(d):\n"
        "    return [dataclasses.replace(synthetic.evaluation(d, cost=c), name=f'c{c}')"
        " for c in (0.001, 0.01)]\n"
    )
    assert cli.main(["--root", str(ledger.root), "run", str(probe)]) == 1
    err = capsys.readouterr().err
    assert err.count("lab run:") == 1
    assert "'<lambda>' is not a module-level function of probe" in err
    assert _types(ledger) == {Event.DATASET: 1}


def test_one_lab_run_scores_every_pipeline_under_each_named_evaluation(
    ledger, dataset, scratch, capsys
):
    sweep = scratch / "sweep.py"
    sweep.write_text(
        "import dataclasses\nfrom tests import synthetic\n"
        "pipelines = synthetic.pipelines[:2]\n"
        "def evaluations(d):\n"
        "    return [dataclasses.replace(synthetic.evaluation(d, cost=c), name=f'c{c}')"
        " for c in (0.001, 0.01)]\n"
    )
    assert cli.main(["--root", str(ledger.root), "run", str(sweep)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"evaluation [0-9a-f]{16} c0\.001\n", out) and " c0.01\n" in out
    rows = ledger.sql(
        "SELECT evaluation_name AS e, COUNT(*) AS n FROM score_latest GROUP BY 1"
    )
    assert {r["e"]: r["n"] for r in rows} == {"c0.001": 2, "c0.01": 2}
    assert _count(ledger, "raw_fit") == 2 * len(
        synthetic.evaluation(dataset).split.folds(load(ledger, dataset))
    )


def test_any_evaluation_field_change_is_a_new_evaluation_id(dataset, evaluation):
    cost = synthetic.evaluation(dataset, cost=0.01)
    one = dataclasses.replace(
        evaluation, split=dataclasses.replace(evaluation.split, horizon=2)
    )
    assert len({evaluation.id, cost.id, one.id}) == 3


def test_the_schedule_is_embargoed_and_stored_once_in_the_evaluation_event(
    ledger, dataset, evaluation
):
    data = load(ledger, dataset)
    folds = evaluation.split.folds(data)
    assert len(folds) >= evaluation.split.min_folds
    for fold in folds:
        skipped = data.ts[fold.train[-1][1] : fold.test[0]]
        assert len(np.unique(skipped)) == evaluation.split.embargo_timestamps
    _run(ledger, evaluation, synthetic.ridge(3))
    stored = ledger.latest(Event.EVALUATION, evaluation.id)["payload"]
    assert (
        len(stored["folds"]) == len(folds)
        and stored["metrics"] == evaluation.directions
    )
    config = ledger.sql("SELECT config FROM raw_evaluation")[0]["config"]
    assert json.loads(config) == {
        "__type__": "tests.synthetic:SimConfig",
        "cost": 0.001,
    }


def test_row_splits_make_disjoint_embargoed_folds(ledger, dataset):
    data = load(ledger, dataset)
    n = data.rows
    folds = splits.BlockedKFold(k=4, embargo_rows=10).folds(data)
    assert [f.label for f in folds] == [f"block {i}" for i in range(4)]
    assert folds[0].train == ((folds[0].test[1] + 10, n),)
    middle = folds[1]
    lo, hi = middle.test
    assert middle.train == ((0, lo - 10), (hi + 10, n))
    assert sum(hi - lo for lo, hi in (f.test for f in folds)) == n
    (hold,) = splits.Holdout(train_fraction=0.8, embargo_rows=5).folds(data)
    cut = data.boundary(round(0.8 * n))
    assert hold.train == ((0, cut),) and hold.test[0] == cut + 5


def test_row_splits_cut_on_timestamp_boundaries():
    stamps = np.datetime64("2000-01-01T00:00:00", "s") + np.arange(10).astype(
        "timedelta64[s]"
    )
    dataset = Dataset({"x": np.zeros(70)}, np.repeat(stamps, 7))
    folds = splits.BlockedKFold(k=3).folds(dataset)
    assert [f.test for f in folds] == [(0, 21), (21, 42), (42, 70)]
    (hold,) = splits.Holdout(0.75).folds(dataset)
    assert hold.train == ((0, 49),) and hold.test == (49, 70)
    (walk,) = splits.WalkForward(45, 70, 20, min_folds=1).folds(dataset)
    assert walk.train == ((0, 42),) and walk.test == (42, 63)
    bare = Dataset({"x": np.zeros(70)})
    assert splits.Holdout(0.75).folds(bare)[0].train == ((0, 52),)


def test_a_kfold_stops_where_holdout_cuts_and_keeps_its_id(ledger, dataset):
    data = load(ledger, dataset)
    folds = splits.BlockedKFold(k=5, train_fraction=0.8).folds(data)
    (hold,) = splits.Holdout(train_fraction=0.8).folds(data)
    ends = [hi for f in folds for _, hi in [*f.train, f.test]]
    assert max(ends) == hold.train[0][1]
    assert identity.content_hash(splits.BlockedKFold(k=5)) == identity.content_hash(
        splits.BlockedKFold(k=5, train_fraction=1.0)
    )


def test_a_clockless_dataset_fits_on_segments_and_refuses_walk_forward(ledger):
    bare = Dataset(
        {
            "f0": np.arange(100.0),
            "f1": np.ones(100),
            "f2": np.zeros(100),
            "ret_1": np.arange(100.0),
        }
    )
    assert bare.matrix(((0, 10), (90, 100)), ("f0", "f1")).shape == (20, 2)
    with pytest.raises(Refused, match="no timestamps; use a row-based split"):
        splits.CalendarWalkForward(first_cutoff="2025-01-01").folds(bare)
    assert synthetic.ridge_fit(
        bare, ((0, 50), (60, 100)), synthetic.RidgeConfig(0, 1.0)
    ).weights.shape == (3,)


def test_row_walk_forward_steps_by_rows(ledger, dataset):
    data = load(ledger, dataset)
    plain = splits.WalkForward(
        first_cutoff_rows=1000, step_rows=500, window_rows=500, embargo_rows=10
    )
    folds = plain.folds(data)
    assert len(folds) == (data.rows - 1000) // 500
    assert folds[0].train == ((0, 990),) and folds[0].test == (1000, 1500)
    assert folds[1].label == "row 1500"
    evaluation = synthetic.evaluation(dataset, split=plain)
    report = _run(ledger, evaluation, synthetic.ridge(0))
    assert report.fits_computed == len(folds) and report.scores_recorded == 1


def test_a_day_walk_forward_steps_over_dates_with_rows():
    days = np.array(
        [
            "2019-06-07",
            "2019-06-10",
            "2019-06-11",
            "2019-06-12",
            "2019-06-13",
            "2019-06-14",
        ],
        "datetime64[D]",
    )
    hours = np.array([9, 15], "timedelta64[h]")
    dataset = Dataset({"x": np.arange(12.0)}, (days[:, None] + hours).ravel())
    folds = splits.CalendarWalkForward("2019-06-08", unit="day", min_folds=1).folds(
        dataset
    )
    assert [f.label for f in folds] == [f"2019-06-1{d}" for d in range(5)]
    assert folds[0].test == (2, 4) and folds[0].train == ((0, 2),)
    assert folds[-1].test == (10, 12)
    with pytest.raises(Refused, match=r"no train rows: \['2019-06-07'\]"):
        splits.CalendarWalkForward("2019-06-07", unit="day", min_folds=1).folds(dataset)
    with pytest.raises(Refused, match="0 months from the first cutoff"):
        splits.CalendarWalkForward("2019-06-12").folds(dataset)


def test_a_calendar_split_stops_at_end_and_embargoes_whole_timestamps():
    d = np.arange(np.datetime64("2021-01-04"), np.datetime64("2022-06-25"))
    d = d[np.is_busday(d)]
    dataset = Dataset({"x": np.arange(d.size, dtype=float)}, d)
    folds = splits.CalendarWalkForward(
        "2021-03-01",
        every=2,
        window=2,
        embargo_timestamps=2,
        end="2021-12-06",
        min_folds=1,
    ).folds(dataset)
    last = max(dataset.ts[f.test[1] - 1] for f in folds)
    assert last < np.datetime64("2021-12-06")
    gaps = [len(np.unique(dataset.ts[f.train[0][1] : f.test[0]])) for f in folds]
    assert gaps == [2] * len(folds) and len(folds) == 4


def test_a_month_split_that_drops_the_tail_says_how_to_score_it(ledger, dataset):
    data = load(ledger, dataset)
    split = splits.CalendarWalkForward(first_cutoff="2025-11-01", horizon=2)
    with pytest.raises(Refused, match="set end=2026-01-01 or later") as info:
        split.folds(data)
    assert "use unit='day'" in str(info.value)


def test_a_later_horizon_scores_the_next_window_and_shares_every_fit(
    ledger, dataset, evaluation
):
    data = load(ledger, dataset)
    later = dataclasses.replace(
        evaluation, split=dataclasses.replace(evaluation.split, horizon=2)
    )
    near, far = evaluation.split.folds(data), later.split.folds(data)
    assert [f.train for f in far] == [f.train for f in near[:-1]]
    assert [f.test for f in far] == [f.test for f in near[1:]]
    _run(ledger, evaluation, synthetic.ridge(3))
    report = _run(ledger, later, synthetic.ridge(3))
    assert report.fits_computed == 0 and report.fits_reused == len(far)
    assert report.predictions_computed == len(far) and report.scores_recorded == 1


def test_the_clock_keeps_nanoseconds_and_the_dataset_id_sees_them():
    t0 = np.datetime64("2019-06-10T09:00:00", "ns")
    a = Dataset({"x": np.zeros(2)}, t0 + np.array([1, 2], "timedelta64[ns]"))
    b = Dataset({"x": np.zeros(2)}, t0 + np.array([1, 3], "timedelta64[ns]"))
    assert data_id(a) != data_id(b)
    assert np.array_equal(rows_load(rows_save(a)).ts, a.ts)


# Identity =============================================================================


def test_a_docstring_edit_keeps_the_code_key_and_a_code_edit_moves_it():
    source = b'def f(x):\n    """Adds one."""\n    return x + 1\n'
    reworded = b'def f(x):\n    """Adds\n    one more."""\n    return x + 1\n'
    assert identity.code_key(source) == identity.code_key(reworded)
    assert identity.code_key(source) == identity.code_key(
        b'def f(x):\n    ""\n    return x + 1\n'
    )
    assert identity.code_key(source) == identity.code_key(
        b"def f(x):\n    return x+1\n"
    )
    assert identity.code_key(source) != identity.code_key(
        b'def f(x):\n    """Adds one."""\n    return x + 2\n'
    )


def test_two_dataclasses_with_one_qualname_in_two_modules_differ():
    source = """
        import dataclasses
        @dataclasses.dataclass(frozen=True)
        class Split:
            rows: int
    """
    a = _module("twin_split_a", source)
    b = _module("twin_split_b", source)
    assert identity.content_hash(a.Split(5)) != identity.content_hash(b.Split(5))


def test_dict_keys_keep_their_type():
    assert identity.content_hash({1: 0.5}) != identity.content_hash({"1": 0.5})
    assert identity.content_hash({1: 0.5}) != identity.content_hash({"int:1": 0.5})
    assert identity.canonical({1: "a", "1": "b"}) == {"1": "b", '["int", 1]': "a"}


def test_signature_annotations_and_import_order_keep_the_code_key():
    source = b"import os\nimport re\n\ndef f(x: int) -> int:\n    return x\n"
    reworded = b"import re\nimport os\n\ndef f(x: 'float') -> float:\n    return x\n"
    assert identity.code_key(source) == identity.code_key(reworded)
    assert identity.code_key(source) != identity.code_key(
        source.replace(b"return x", b"return x + 1")
    )


def test_a_default_the_memo_cannot_see_stays_in_the_id():
    source = """
        import dataclasses
        @dataclasses.dataclass(frozen=True)
        class Knob:
            scale: float = 1.0
    """
    knob = _module("sourceless_knob", source).Knob()
    assert identity.canonical(knob) == {
        "__type__": "sourceless_knob:Knob",
        "scale": 1.0,
    }
    assert identity.canonical(synthetic.BlendConfig()) == {
        "__type__": "tests.synthetic:BlendConfig"
    }


def test_a_reached_module_the_memo_cannot_see_is_refused(tmp_path, monkeypatch):
    for directory in ("lib", "root"):
        (tmp_path / directory).mkdir()
        monkeypatch.syspath_prepend(tmp_path / directory)
    (tmp_path / "lib" / "far_lib.py").write_text("K = 1\n")
    (tmp_path / "root" / "near_steps.py").write_text(
        "import numpy\nimport far_lib\n\ndef fit(dataset, train, config):\n"
        "    return far_lib.K\n"
    )
    fit = importlib.import_module("near_steps").fit
    with pytest.raises(
        Refused, match=r"far_lib.py is reached .* outside the code root"
    ):
        identity.refuse_unseen_code((fit,), tmp_path / "root")
    identity.refuse_unseen_code((fit,), tmp_path)
    source = tmp_path / "root" / "stale_steps.py"
    source.write_text("def fit(dataset, train, config):\n    return +1.0\n")
    py_compile.compile(str(source), importlib.util.cache_from_source(str(source)))
    stat = source.stat()
    source.write_text("def fit(dataset, train, config):\n    return -1.0\n")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    stale = importlib.import_module("stale_steps").fit
    assert stale(None, None, None) == 1.0
    with pytest.raises(Refused, match="ran from bytecode .* that is not its source"):
        identity.refuse_unseen_code((stale,), tmp_path)


def test_a_numpy_bool_is_a_bool():
    assert identity.canonical(np.bool_(True)) is True


def test_a_module_imported_for_a_constant_is_in_the_closure(tmp_path, monkeypatch):
    package = tmp_path / "const_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "knobs.py").write_text("SCALE = 2.0\n")
    (package / "model.py").write_text(
        "from .knobs import SCALE\ndef fit(dataset, train, config):\n    return SCALE\n"
    )
    (package / "other.py").write_text("from .knobs import SCALE\n")
    monkeypatch.syspath_prepend(tmp_path)
    model = importlib.import_module("const_pkg.model")
    importlib.import_module("const_pkg.other")
    shas = identity.import_shas((model.fit,), tmp_path)
    assert "const_pkg/knobs.py" in shas and "const_pkg/other.py" not in shas


def test_a_namespace_package_submodule_is_in_the_closure(tmp_path, monkeypatch):
    (tmp_path / "ns_pkg").mkdir()
    (tmp_path / "ns_pkg" / "helpers.py").write_text("def two():\n    return 2\n")
    (tmp_path / "ns_pkg_model.py").write_text(
        "import ns_pkg.helpers\n"
        "def fit(dataset, train, config):\n"
        "    return ns_pkg.helpers.two()\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    model = importlib.import_module("ns_pkg_model")
    assert "ns_pkg/helpers.py" in identity.import_shas((model.fit,), tmp_path)


def test_an_editable_single_module_install_hashes_its_source(tmp_path, monkeypatch):
    source = _editable(tmp_path, monkeypatch, "single_tool", single=True)
    fit = importlib.import_module("single_tool_steps").fit
    before = identity.imported_dists((fit,), tmp_path / "study")["single_tool"]
    source.write_text("SCALE = 2.0\n")
    identity._editable_source.cache_clear()
    assert identity.imported_dists((fit,), tmp_path / "study")["single_tool"] != before


def test_an_editable_lock_covers_the_modules_the_steps_reach(tmp_path, monkeypatch):
    package = _editable(tmp_path, monkeypatch, "lock_tool")
    fit = importlib.import_module("lock_tool_steps").fit

    def lock() -> str:
        identity._editable_source.cache_clear()
        return identity.imported_dists((fit,), tmp_path / "study")[package.name]

    before = lock()
    (package / "b.py").write_text("def main():\n    return 1\n")
    assert lock() == before
    (package / "a.py").write_text("SCALE = 2.0\n")
    assert lock() != before


def test_one_process_fits_under_one_lock_and_records_the_tool_commit(
    ledger, tmp_path, monkeypatch, scratch
):
    package = _editable(tmp_path, monkeypatch, "run_tool", study=scratch)
    fit = importlib.import_module("run_tool_steps").fit
    head = subprocess.run(
        ["git", "-C", package.parent, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (package / "b.py").write_text("def main():\n    return 1\n")
    evaluation = synthetic.evaluation(synthetic.dataset(ledger))
    pipeline = dataclasses.replace(synthetic.ridge(1), fit=fit, name="tool")

    def run() -> runs.RunReport:
        return runs.run(ledger, [pipeline], evaluation, code_root=REPO)

    first = run()
    assert not first.failed and first.fits_computed > 0
    (package / "a.py").write_text("SCALE = 2.0\n")
    assert run().fits_computed == 0
    assert len(ledger.sql("SELECT DISTINCT env_lock FROM raw_fit")) == 1
    started = ledger.latest(Event.RUN, first.run)["payload"]["editable"]
    assert started[package.name]["commit"] == head
    assert started[package.name]["dirty"] is True
    editable = ledger.sql("SELECT editable FROM raw_run")[0]["editable"]
    assert json.loads(editable)[package.name]["commit"] == head


# Dataset ==============================================================================


def test_feature_columns_are_read_from_their_store_not_memory(tmp_path):
    rows = synthetic.generate(
        start="2025-01-01", months=1, rows_per_day=4, seed=1, drift_at="2025-01-20"
    )
    store = tmp_path / "f.parquet"
    lag = np.arange(rows.rows, dtype=np.int32)
    pl.DataFrame({"f0_lag": lag}).write_parquet(store)
    dataset = Dataset(rows.columns, rows.ts, ("f0_lag",), {"f0_lag": store})
    assert "f0_lag" not in dataset.columns
    assert dataset.column("f0_lag", (3, 7)).tolist() == [3, 4, 5, 6]
    assert dataset.column("f0_lag", ((0, 2), (5, 6))).tolist() == [0, 1, 5]
    matrix = dataset.matrix((2, 5), ("f0", "f0_lag"))
    assert matrix.dtype == np.float64 and matrix[:, 1].tolist() == [2.0, 3.0, 4.0]
    view = dataset.upto(10)
    assert view.rows == 10 and view.column("f0_lag", (8, 10)).tolist() == [8, 9]
    with pytest.raises(ValueError, match="visible before the cutoff"):
        view.column("f0_lag", (8, 11))
    with pytest.raises(KeyError, match=r"feature columns \['f0_lag'\]"):
        dataset.column("f9", (0, 1))


def test_a_fit_view_blanks_every_column_outside_its_train_segments(tmp_path):
    store = tmp_path / "f.parquet"
    pl.DataFrame({"lag": np.arange(6, dtype=np.int32)}).write_parquet(store)
    clock = np.datetime64("2025-01-01", "ns") + np.arange(6).astype("timedelta64[h]")
    columns = {"x": np.arange(6), "s": _strings(*"abcdef"), "t": clock}
    full = Dataset(columns, clock, ("lag",), {"lag": store})
    view = full.train_view(((0, 2), (4, 6)))
    assert np.isnan(view.columns["x"][2:4]).all() and view.columns["x"][4] == 4.0
    assert view.columns["s"].tolist() == ["a", "b", None, None, "e", "f"]
    assert np.isnat(view.columns["t"][2:4]).all() and view.ts[3] == clock[3]
    lag = view.matrix((0, 6), ("x", "lag"))[:, 1]
    assert np.isnan(lag[2:4]).all() and lag[5] == 5.0
    assert np.isnan(view.upto(3).column("lag", (2, 3))).all()


@pytest.mark.parametrize("segment", [(-1, 0), (-3, 10), (5, 4)])
def test_a_segment_below_row_zero_or_reversed_is_refused(tmp_path, segment):
    store = tmp_path / "f.parquet"
    pl.DataFrame({"f": np.arange(100.0)}).write_parquet(store)
    view = Dataset({"x": np.zeros(100)}, None, ("f",), {"f": store}).upto(10)
    for name in ("x", "f"):
        with pytest.raises(ValueError, match=rf"segment \({segment[0]}, "):
            view.column(name, segment)


def test_timestamps_with_a_nat_are_refused():
    ts = np.array(["2025-01-01", "NaT"], dtype="datetime64[s]")
    with pytest.raises(ValueError, match="NaT"):
        Dataset({"x": np.zeros(2)}, ts)


# Run ==================================================================================


def test_a_run_posts_fits_predictions_and_one_score_and_a_rerun_writes_nothing(
    ledger, dataset, evaluation
):
    folds = evaluation.split.folds(load(ledger, dataset))
    first = _run(ledger, evaluation, synthetic.ridge(3))
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds)
    assert first.scores_recorded == 1
    before = len(ledger.events())
    second = _run(ledger, evaluation, synthetic.ridge(3))
    assert (
        second.fits_computed,
        second.predictions_computed,
        second.scores_recorded,
    ) == (0, 0, 0)
    assert (second.fits_reused, second.predictions_reused, second.scores_reused) == (
        len(folds),
        len(folds),
        1,
    )
    assert second.run == "" and ledger.events()[before:] == []


def test_adding_a_pipeline_costs_only_its_own_fits(ledger, dataset, evaluation):
    folds = evaluation.split.folds(load(ledger, dataset))
    _run(ledger, evaluation, synthetic.ridge(3))
    report = _run(ledger, evaluation, synthetic.ridge(3), synthetic.ridge(6))
    assert report.fits_computed == len(folds) and report.scores_recorded == 1
    assert report.fits_reused == len(folds) and report.scores_reused == 1
    assert len(ledger.events(Event.RUN)) == 2


def test_steps_see_only_the_rows_before_their_cutoff(ledger, dataset, evaluation):
    folds = evaluation.split.folds(load(ledger, dataset))
    synthetic.SEEN_ROWS.clear()
    peeking = dataclasses.replace(synthetic.ridge(1), fit=synthetic.peeking_fit)
    _run(ledger, evaluation, peeking)
    assert synthetic.SEEN_ROWS == [fold.train[-1][1] for fold in folds]
    first = load(ledger, dataset).upto(10)
    assert first.rows == 10 and first.column("f0", (8, 10)).shape == (2,)
    assert first.column("f0", (np.int64(8), np.int64(10))).shape == (2,)
    with pytest.raises(ValueError, match="before"):
        first.matrix((8, 12), ("f0",))


def test_targets_are_hidden_inside_the_test_range(ledger, dataset, evaluation):
    cheat = dataclasses.replace(synthetic.ridge(1), predict=synthetic.cheating_predict)
    error = _run(ledger, evaluation, cheat).failed["ridge_1m"]
    assert re.search(r"non-finite value on (\d+) of \1 rows of fold 2025-05-01", error)
    assert not ledger.events(Event.PREDICTIONS)


def memorised_f0_fit(dataset: Dataset, train, config) -> synthetic.RidgeModel:
    return synthetic.RidgeModel(dataset.column("f0", (0, dataset.rows)), 0.0)


def rows_known_predict(model, dataset: Dataset, rng: Range) -> np.ndarray:
    """How many f0 values the view shows before the range."""
    known = np.isfinite(dataset.column("f0", (0, rng[0]))).sum()
    return np.full(rng[1] - rng[0], float(known))


def test_a_fit_and_an_in_sample_member_see_no_column_outside_the_train_segments(
    ledger, dataset
):
    evaluation = synthetic.evaluation(dataset, split=splits.BlockedKFold(k=3))
    (lo, hi) = evaluation.split.folds(load(ledger, dataset))[1].test
    memo = dataclasses.replace(
        synthetic.ridge(0), fit=memorised_f0_fit, predict=rows_known_predict
    )
    assert not _run(ledger, evaluation, synthetic.blend(memo)).failed
    (fit,) = ledger.sql(
        "SELECT model FROM raw_fit WHERE pipeline = ? AND label = 'block 1'",
        (memo.id,),
    )
    f0 = memo.load(ledger.get_blob(fit["model"])).weights
    assert np.isnan(f0[lo:hi]).all() and not np.isnan(np.r_[f0[:lo], f0[hi:]]).any()
    (member,) = ledger.sql(
        "SELECT id FROM raw_prediction WHERE fold = 1 AND range_start = ?", (hi,)
    )
    assert (_load_predictions(ledger, member["id"]) == lo).all()


def test_a_kfold_fit_that_memorizes_labels_replays_none(ledger, dataset):
    evaluation = synthetic.evaluation(dataset, split=splits.BlockedKFold(k=3))
    peek = dataclasses.replace(
        synthetic.ridge(0),
        name="peek",
        fit=synthetic.label_memory_fit,
        predict=synthetic.label_replay_predict,
    )
    assert not _run(ledger, evaluation, peek).failed
    for e in ledger.events(Event.PREDICTIONS):
        assert not np.nan_to_num(_load_predictions(ledger, e["id"])).any()


def test_a_fold_training_on_an_unrevealed_label_is_refused(ledger):
    dataset = synthetic.panel(
        ledger, reveal={synthetic.TARGET: datetime.timedelta(minutes=2)}
    )
    bare = splits.WalkForward(200, 50, 50, min_folds=2)
    evaluation = synthetic.evaluation(dataset, split=bare)
    for dry in (True, False):
        with pytest.raises(
            Refused,
            match=r"4 folds train.*rows 198 to 199 whose ret_1 .*0:02:00.*embargo 1 "
            r"more timestamps \(2 rows\)",
        ):
            _run(ledger, evaluation, synthetic.ridge(0), dry=dry)
    assert not ledger.events(Event.EVALUATION)


@dataclasses.dataclass(frozen=True)
class OverlapSplit:
    def folds(self, dataset):
        return [splits.Fold("f0", ((0, 3000),), (2500, 3500))]


@pytest.mark.parametrize("dry", [False, True])
def test_a_fold_whose_train_overlaps_its_test_range_is_refused(ledger, dataset, dry):
    evaluation = synthetic.evaluation(dataset, split=OverlapSplit())
    with pytest.raises(
        Refused,
        match=r"fold f0 trains on segment \(0, 3000\), which overlaps its "
        r"test range \(2500, 3500\)",
    ):
        _run(ledger, evaluation, synthetic.ridge(0), dry=dry)
    assert not ledger.events(Event.EVALUATION)


def test_a_predict_that_reads_the_future_inside_its_range_is_refused(
    ledger, dataset, evaluation
):
    peek = dataclasses.replace(synthetic.ridge(1), predict=synthetic.peeking_predict)
    report = _run(ledger, evaluation, peek)
    assert "predict read past row" in report.failed["ridge_1m"]
    assert ledger.events(Event.SCORE) == []


def day_mean_predict(model, dataset: Dataset, rng: Range) -> np.ndarray:
    """Each row's f0 mean over its date in the range: reads later rows of the date."""
    day = dataset.ts[rng[0] : rng[1]].astype("datetime64[D]")
    _, at = np.unique(day, return_inverse=True)
    x = dataset.column("f0", rng)
    return (np.bincount(at, x) / np.bincount(at))[at]


def test_the_probes_cut_inside_a_date_however_the_calendar_falls(ledger):
    rows = synthetic.generate(
        start="2025-01-01", months=3, rows_per_day=20, seed=7, drift_at="2025-02-01"
    )
    dataset = record(
        ledger, rows, source="s", params={}, filters=(), targets=(synthetic.TARGET,)
    )
    two_days = splits.CalendarWalkForward(
        "2025-02-01", unit="day", every=20, window=2, embargo_timestamps=1
    )
    evaluation = synthetic.evaluation(dataset, split=two_days)
    sixths = synthetic.featured(synthetic.ridge(1), synthetic.day_mean, "sixths")
    window = dataclasses.replace(
        synthetic.ridge(1), predict=day_mean_predict, name="window"
    )
    report = _run(ledger, evaluation, sixths, window)
    assert "features read past row 310" in report.failed["sixths"]
    assert "predict read past row" in report.failed["window"]


def test_rounding_passes_the_probe_and_a_leak_is_sized(ledger, evaluation):
    full = np.array([1.0, -2.0, 4.0, np.nan])
    runs._refuse_changed("x", "prediction", full, full[:3] + 1e-10, at=3)
    runs._refuse_changed("x", "prediction", full, full, at=None)
    with pytest.raises(Refused, match="3 of 3 rows moved") as info:
        runs._refuse_changed("x", "prediction", full, full[:3] + 1e-6, at=3, first=10)
    assert "at row 10" in str(info.value) and "read past row 3" in str(info.value)
    assert "2.5e-07 of the largest |prediction| 4" in str(info.value)
    with pytest.raises(Refused, match="is not stable"):
        runs._refuse_changed("x", "prediction", full, np.array([1.0, -2.0, 4.0, 0.0]))
    with pytest.raises(Refused, match="NaN|nan"):
        runs._refuse_changed("x", "prediction", full, np.array([1.0, -2.0, 4.0, 0.0]))
    wobbly = dataclasses.replace(synthetic.ridge(1), predict=synthetic.jittery_predict)
    report = _run(ledger, evaluation, wobbly)
    assert not report.failed and "ridge_1m" in _latest(ledger, evaluation)


def scalar_predict(model, dataset: Dataset, rng) -> float:
    return 0.0


def first_row_postprocess(pred: np.ndarray, dataset: Dataset, rng, config):
    return pred[:1]


def two_column_predict(model, dataset: Dataset, rng) -> np.ndarray:
    one = synthetic.ridge_predict(model, dataset, rng)
    return np.column_stack([one, -one])


def widening_predict(model, dataset: Dataset, rng) -> np.ndarray:
    one = synthetic.ridge_predict(model, dataset, rng)
    return one if rng[0] < 3000 else np.column_stack([one, one])


def member_sum_predict(model, dataset: Dataset, rng, members: list) -> np.ndarray:
    (only,) = members
    assert only.shape == (rng[1] - rng[0], 2)
    return only


def column_gap(pred: np.ndarray, dataset: Dataset, rng, config) -> np.ndarray:
    assert pred.shape == (rng[1] - rng[0], 2)
    return pred[:, 0] - pred[:, 1]


gap_scorer = Scorer(
    column_gap, metrics=lambda s: {"gap": float(np.mean(s))}, directions={"gap": "max"}
)


def test_a_predict_of_k_columns_is_stored_scored_and_blended_as_such(ledger, dataset):
    wide = dataclasses.replace(synthetic.ridge(1), predict=two_column_predict)
    blend = dataclasses.replace(
        synthetic.blend(wide), predict=member_sum_predict, name="column_blend"
    )
    evaluation = dataclasses.replace(synthetic.evaluation(dataset), scorer=gap_scorer)
    report = _run(ledger, evaluation, wide, blend)
    assert report.failed == {} and report.scores_recorded == 2
    first = ledger.sql("SELECT id, range_start, range_end FROM raw_prediction")[0]
    stored = _load_predictions(ledger, first["id"])
    assert stored.shape == (first["range_end"] - first["range_start"], 2)
    sha = ledger.sql("SELECT blob FROM raw_prediction WHERE id = ?", (first["id"],))
    frame = pl.read_parquet(io.BytesIO(ledger.get_blob(sha[0]["blob"])))
    assert frame.columns == ["prediction_0", "prediction_1"]
    assert ledger.sql("SELECT value FROM board WHERE name = 'ridge_1m'")[0]["value"] > 0


def test_a_predict_whose_width_changes_between_folds_is_refused(ledger, evaluation):
    p = dataclasses.replace(synthetic.ridge(1), predict=widening_predict, name="wide")
    report = _run(ledger, evaluation, p)
    assert "the width is a property of predict" in report.failed["wide"]


@pytest.mark.parametrize(
    "change", [{"predict": scalar_predict}, {"postprocess": first_row_postprocess}]
)
def test_an_output_that_is_not_one_row_per_range_row_is_refused(
    ledger, evaluation, change
):
    p = dataclasses.replace(synthetic.ridge(1), **change, name="shaped")
    report = _run(ledger, evaluation, p)
    assert f"{next(iter(change))} returned shape" in report.failed["shaped"]
    assert "one row per row of the range" in report.failed["shaped"]
    rows = ledger.sql("SELECT id, range_start, range_end FROM raw_prediction")
    assert all(
        len(_load_predictions(ledger, r["id"])) == r["range_end"] - r["range_start"]
        for r in rows
    )


def test_a_module_edited_during_the_run_is_refused_not_recorded(ledger, tmp_path):
    code = tmp_path / "steps_e.py"
    code.write_text(
        (REPO / "tests" / "synthetic.py").read_text().replace("SYN", "SYNE")
    )
    module = cli._load(str(code))
    evaluation = module.evaluation(module.dataset(ledger))
    editing = dataclasses.replace(module.ridge(1), fit=module.self_editing_fit)
    report = runs.run(ledger, [editing], evaluation, code_root=tmp_path)
    assert "source changed during the run" in report.failed["ridge_1m"]
    assert ledger.events(Event.FIT) == []


def test_a_feature_step_runs_once_for_two_pipelines_and_a_lookahead_is_refused(
    ledger, evaluation
):
    synthetic.FEATURE_CALLS.clear()
    a = synthetic.featured(synthetic.ridge(1))
    b = synthetic.featured(synthetic.ridge(3))
    report = _run(ledger, evaluation, a, b)
    n = load(ledger, evaluation.dataset).rows
    sixths = [j * n // 6 for j in range(1, 6)]
    cuts = sorted({*sixths, *(row // 20 * 20 + 10 for row in sixths)})
    assert not report.failed and synthetic.FEATURE_CALLS[1:] == cuts
    rows = ledger.sql("SELECT columns, probe_rows FROM raw_feature")
    assert len(rows) == 1 and json.loads(rows[0]["columns"]) == ["f0_lag"]
    assert json.loads(rows[0]["probe_rows"]) == cuts
    assert _count(ledger, "raw_fit") == report.fits_computed
    assert set(_latest(ledger, evaluation)) == {"ridge_1m_lag", "ridge_3m_lag"}
    assert a.id != synthetic.ridge(1).id
    leaky = synthetic.featured(synthetic.ridge(1), synthetic.next_f0, "leaky")
    copier = synthetic.featured(synthetic.ridge(1), synthetic.target_copy, "copier")
    report = _run(ledger, evaluation, leaky, copier)
    assert "features read past row" in report.failed["leaky"]
    assert synthetic.TARGET in report.failed["copier"]
    assert _count(ledger, "raw_feature") == 1


def test_a_features_tuple_is_one_event_per_step_shared_with_single_step_pipelines(
    ledger, evaluation
):
    both = synthetic.featured(
        synthetic.ridge(3), (synthetic.lagged_f0, synthetic.lagged_f1), "both"
    )
    both = dataclasses.replace(
        both,
        config=dataclasses.replace(
            both.config, columns=synthetic.FEATURES + ("f0_lag", "f1_lag")
        ),
    )
    assert both.feature_functions == (synthetic.lagged_f0, synthetic.lagged_f1)
    report = _run(ledger, evaluation, both, synthetic.featured(synthetic.ridge(1)))
    assert not report.failed
    rows = ledger.sql("SELECT columns FROM raw_feature ORDER BY seq")
    assert [json.loads(r["columns"]) for r in rows] == [["f0_lag"], ["f1_lag"]]
    weights = {
        r["pipeline"]: both.load(ledger.get_blob(r["model"])).weights.shape
        for r in ledger.sql("SELECT pipeline, model FROM raw_fit")
    }
    assert weights[both.id] == (5,)
    features = {r["features"] for r in ledger.sql("SELECT features FROM raw_fit")}
    assert len(features) == 2


def other_f0_lag(dataset: Dataset) -> dict[str, np.ndarray]:
    return {"f0_lag": np.full(dataset.rows, 7.0)}


@pytest.mark.parametrize("second", [other_f0_lag, synthetic.lagged_f0])
def test_feature_steps_naming_one_column_are_refused(ledger, evaluation, second):
    p = synthetic.featured(synthetic.ridge(1), (synthetic.lagged_f0, second), "clash")
    report = _run(ledger, evaluation, p)
    assert "name the same columns ['f0_lag']" in report.failed["clash"]
    assert _count(ledger, "raw_fit") == 0


def prefix_named_features(dataset: Dataset) -> dict[str, np.ndarray]:
    """Names its column by whether it sees more than 7000 rows."""
    name = "whole" if dataset.rows > 7000 else "part"
    return {name: dataset.columns["f0"].copy()}


def test_a_feature_step_whose_names_change_with_the_prefix_is_refused(
    ledger, evaluation
):
    p = dataclasses.replace(
        synthetic.ridge(1), features=prefix_named_features, name="shifty"
    )
    report = _run(ledger, evaluation, p)
    assert report.failed["shifty"].startswith("Refused: shifty: features name columns")
    assert "['whole']" in report.failed["shifty"]
    assert "['part'] on the rows before" in report.failed["shifty"]


def test_a_dry_run_names_the_moved_module_and_writes_nothing(ledger, tmp_path):
    code = tmp_path / "steps_d.py"
    code.write_text((REPO / "tests" / "synthetic.py").read_text())
    module = cli._load(str(code))
    evaluation = module.evaluation(module.dataset(ledger))
    runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    code.write_text(code.read_text().replace("0.01 * np.sum", "0.02 * np.sum"))
    seq = ledger.sql("SELECT max(seq) AS m FROM event")[0]["m"]
    blobs, lines = set(ledger.blobs.iterdir()), []
    report = runs.run(
        ledger,
        [module.ridge(3), module.featured(module.ridge(3))],
        evaluation,
        code_root=tmp_path,
        dry=True,
        log=lines.append,
    )
    folds = len(evaluation.split.folds(load(ledger, evaluation.dataset)))
    assert report.run == "" and report.fits_computed == 2 * folds
    assert report.scores_recorded == 2 and not report.failed
    would = [line for line in lines if line.startswith("would fit")]
    assert len(would) == 2 * folds
    plain = [line for line in would if "ridge_3m " in line]
    assert len(plain) == folds and all(line.endswith(": steps_d.py") for line in plain)
    lag = [line for line in would if "ridge_3m_lag" in line]
    assert len(lag) == folds and all(
        line.endswith("no earlier fit of this pipeline and label") for line in lag
    )
    assert ledger.sql("SELECT max(seq) AS m FROM event")[0]["m"] == seq
    assert set(ledger.blobs.iterdir()) == blobs
    assert _count(ledger, "raw_feature") == 0


def test_a_postprocess_shares_the_fit_and_is_its_own_prediction(
    ledger, dataset, evaluation
):
    folds = evaluation.split.folds(load(ledger, dataset))
    plain = synthetic.ridge(1)
    doubled = dataclasses.replace(
        plain, postprocess=synthetic.scale, postprocess_config=2.0, name="ridge_1m_x2"
    )
    assert doubled.id != plain.id
    report = _run(ledger, evaluation, plain, doubled)
    assert report.fits_computed == len(folds) and report.fits_reused == len(folds)
    assert report.predictions_computed == 2 * len(folds)
    assert report.predictions_reused == 0 and report.scores_recorded == 2
    post = [e for e in ledger.events(Event.PREDICTIONS) if "raw" in e["payload"]]
    assert len(post) == len(folds)
    assert post[0]["payload"]["postprocess_config"] == 2.0
    raw = _load_predictions(ledger, post[0]["payload"]["raw"])
    assert np.allclose(_load_predictions(ledger, post[0]["id"]), 2 * raw)
    again = _run(ledger, evaluation, plain, doubled)
    assert again.run == "" and again.predictions_reused == 2 * len(folds)


def test_a_postprocess_config_enters_the_prediction_id_and_shares_the_fit(
    ledger, evaluation
):
    doubled = dataclasses.replace(
        synthetic.ridge(1),
        postprocess=synthetic.scale,
        postprocess_config=2.0,
        name="x2",
    )
    tripled = dataclasses.replace(doubled, postprocess_config=3.0, name="x3")
    assert tripled.id != doubled.id
    first = _run(ledger, evaluation, doubled)
    report = _run(ledger, evaluation, tripled)
    assert report.fits_computed == 0 and report.fits_reused == first.fits_computed
    assert report.predictions_computed == first.predictions_computed // 2
    post = [e for e in ledger.events(Event.PREDICTIONS) if "raw" in e["payload"]]
    by_config = {e["payload"]["postprocess_config"]: e for e in post}
    assert len(post) == 2 * report.predictions_computed and set(by_config) == {2, 3}
    tripled_values = _load_predictions(ledger, by_config[3.0]["id"])
    raw = _load_predictions(ledger, by_config[3.0]["payload"]["raw"])
    assert np.allclose(tripled_values, 3 * raw)
    assert len({e["payload"]["raw"] for e in post}) == report.predictions_computed


def test_a_blend_reuses_its_members_fits_and_predictions(ledger, dataset, evaluation):
    folds = evaluation.split.folds(load(ledger, dataset))
    one, three = synthetic.ridge(1), synthetic.ridge(3)
    _run(ledger, evaluation, one, three)
    report = _run(ledger, evaluation, synthetic.blend(one, three))
    assert report.fits_reused == 2 * len(folds) and report.fits_computed == len(folds)
    assert report.predictions_reused == 2 * len(folds)
    assert report.predictions_computed == 3 * len(folds)
    assert report.scores_recorded == 1
    blended = [e for e in ledger.events(Event.PREDICTIONS) if "members" in e["payload"]]
    first = blended[0]
    parts = [_load_predictions(ledger, p) for p in first["payload"]["members"]]
    assert np.allclose(_load_predictions(ledger, first["id"]), np.mean(parts, axis=0))
    fit = ledger.latest(Event.FIT, first["payload"]["fit"])
    assert len(fit["payload"]["members"]) == 2
    stands_on = _count(
        ledger, "score_fit WHERE pipeline = ?", synthetic.blend(one, three).id
    )
    assert stands_on == 3 * len(folds)
    assert _run(ledger, evaluation, synthetic.blend(one, three)).run == ""
    assert _run(
        ledger, evaluation, synthetic.blend(one, three, shrink=0.5)
    ).fits_computed == len(folds)


def test_a_member_postprocess_change_refits_an_in_sample_blend(ledger, evaluation):
    member = synthetic.ridge(1)
    first = _run(ledger, evaluation, synthetic.blend(member))
    shifted = dataclasses.replace(
        member, postprocess=synthetic.scale, postprocess_config=2.0, name="x2"
    )
    second = _run(ledger, evaluation, synthetic.blend(shifted))
    assert not first.failed and not second.failed
    assert second.fits_computed == first.fits_computed // 2 > 0


def test_every_fit_of_a_nested_blend_names_a_declared_pipeline(ledger, evaluation):
    report = _run(ledger, evaluation, _nested(synthetic.ridge(1)))
    assert not report.failed, report.failed
    undeclared = ledger.sql(
        "SELECT DISTINCT f.pipeline FROM raw_fit f LEFT JOIN raw_pipeline p "
        "ON p.id = f.pipeline WHERE p.id IS NULL"
    )
    assert not undeclared and _count(ledger, "raw_pipeline") == 5


def test_a_member_listed_twice_is_fit_once(ledger, evaluation):
    r = synthetic.ridge(1)
    blend = dataclasses.replace(
        synthetic.blend(r, r), fit=synthetic.equal_fit, in_sample=False
    )
    report = _run(ledger, evaluation, blend)
    assert not report.failed, report.failed
    assert report.fits_computed == _count(ledger, "raw_fit")


def test_a_blend_sees_its_members_shared_feature_columns(ledger, evaluation):
    a = synthetic.featured(synthetic.ridge(1))
    b = synthetic.featured(synthetic.ridge(3))
    shifted = dataclasses.replace(
        synthetic.blend(a, b),
        fit=synthetic.equal_fit,
        in_sample=False,
        postprocess=synthetic.add_f0_lag,
        name="shifted",
    )
    report = _run(ledger, evaluation, shifted)
    assert not report.failed and shifted.feature_functions == ()
    assert _count(ledger, "raw_feature") == 1
    mixed = dataclasses.replace(shifted, members=(a, synthetic.ridge(3)), name="mixed")
    report = _run(ledger, evaluation, mixed)
    assert "feature columns []" in report.failed["mixed"]


def test_pickle_is_marked_not_portable_and_an_unknown_format_is_refused(
    ledger, dataset, evaluation
):
    pickled = dataclasses.replace(
        synthetic.ridge(1),
        save=synthetic.pickle_save,
        load=synthetic.pickle_load,
        format=formats.Format.PICKLE,
    )
    _run(ledger, evaluation, pickled)
    assert ledger.sql("SELECT portable FROM raw_fit")[0]["portable"] == 0
    with pytest.raises(Refused, match="'tar', not one of"):
        _run(ledger, evaluation, dataclasses.replace(pickled, format="tar"))


def test_a_zip_model_is_portable_only_if_every_member_is(ledger, evaluation):
    zipped = dataclasses.replace(
        synthetic.ridge(1),
        save=synthetic.zip_save,
        load=synthetic.zip_load,
        format=formats.Format.ZIP,
    )
    pickled = dataclasses.replace(
        zipped,
        save=synthetic.zip_pickle_save,
        load=synthetic.zip_pickle_load,
        name="zp",
    )
    _run(ledger, evaluation, zipped, pickled)
    rows = ledger.sql("SELECT pipeline, format, portable, model FROM raw_fit")
    assert {(r["pipeline"], r["format"], r["portable"]) for r in rows} == {
        (zipped.id, "zip", 1),
        (pickled.id, "zip", 0),
    }
    sha = next(r["model"] for r in rows if r["pipeline"] == zipped.id)
    assert zipped.load(ledger.get_blob(sha)).weights.shape == (3,)
    assert formats.zip_load(ledger.get_blob(sha))["note.txt"] == b"ridge"


def test_a_formatter_pass_keeps_every_fit_and_a_code_edit_refits(ledger, tmp_path):
    code = tmp_path / "steps_f.py"
    code.write_text((REPO / "tests" / "synthetic.py").read_text())
    module = cli._load(str(code))
    evaluation = module.evaluation(module.dataset(ledger))
    runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    sha = ledger.events(Event.FIT)[0]["payload"]["import_shas"]["steps_f.py"]
    code.write_text(
        code.read_text()
        .replace('"""Yesterday', '"""   Yesterday')
        .replace("Xc = X - x_mean", "Xc = (X - x_mean)  # centred")
        + "\n# noop\n"
    )
    report = runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    assert report.run == "" and report.fits_computed == report.predictions_computed == 0
    assert ledger.events(Event.FIT)[0]["payload"]["import_shas"]["steps_f.py"] == sha
    code.write_text(
        code.read_text().replace("Xc = (X - x_mean)", "Xc = X - 2 * x_mean")
    )
    module = importlib.reload(module)
    evaluation = module.evaluation(evaluation.dataset)
    assert (
        runs.run(
            ledger, [module.ridge(3)], evaluation, code_root=tmp_path
        ).fits_computed
        > 0
    )


def test_a_feature_edit_that_keeps_the_columns_keeps_every_fit(
    ledger, evaluation, scratch
):
    code = scratch / "feat_k.py"
    code.write_text(
        "import numpy as np\n\n\n"
        "def lag(dataset):\n    f0 = dataset.columns['f0']\n"
        "    return {'f0_lag': np.concatenate([[np.nan], f0[:-1]])}\n"
    )
    module = cli._load(str(code))
    featured = synthetic.featured(synthetic.ridge(3), module.lag)
    _run(ledger, evaluation, featured)
    code.write_text(
        code.read_text().replace(
            "np.concatenate([[np.nan], f0[:-1]])", "np.r_[np.nan, f0[:-1]]"
        )
    )
    module = importlib.reload(module)
    featured = synthetic.featured(synthetic.ridge(3), module.lag)
    report = _run(ledger, evaluation, featured)
    assert _count(ledger, "raw_feature") == 2
    assert report.fits_computed == report.predictions_computed == 0
    rows = ledger.sql("SELECT DISTINCT features FROM raw_fit")
    assert len(rows) == 1 and rows[0]["features"]


def test_fit_and_predict_read_the_feature_columns_in_step_order(ledger, evaluation):
    a = synthetic.featured(synthetic.ridge(3))
    _run(ledger, evaluation, a, synthetic.ridge(1))
    weights = {
        r["pipeline"]: a.load(ledger.get_blob(r["model"])).weights.shape
        for r in ledger.sql("SELECT pipeline, model FROM raw_fit")
    }
    assert weights[a.id] == (4,) and weights[synthetic.ridge(1).id] == (3,)


def test_fits_land_on_the_dataset_stream_and_scores_on_the_evaluation_stream(
    ledger, dataset, evaluation
):
    _run(ledger, evaluation, synthetic.ridge(3))
    assert {e["stream"] for e in ledger.events(Event.FIT)} == {dataset}
    assert {e["stream"] for e in ledger.events(Event.PREDICTIONS)} == {dataset}
    assert {e["stream"] for e in ledger.events(Event.SCORE)} == {evaluation.id}


def test_a_scoring_change_refits_nothing_and_writes_a_new_score(
    ledger, dataset, evaluation
):
    _run(ledger, evaluation, synthetic.ridge(3))
    report = _run(ledger, synthetic.evaluation(dataset, cost=0.01), synthetic.ridge(3))
    assert report.fits_computed == 0 and report.scores_recorded == 1
    assert len(ledger.events(Event.SCORE)) == 2
    directions = {"pnl": "min", "turnover": "min"}
    scorer = dataclasses.replace(evaluation.scorer, directions=directions)
    flipped = dataclasses.replace(evaluation, scorer=scorer)
    report = _run(ledger, flipped, synthetic.ridge(3))
    assert flipped.id == evaluation.id and report.scores_recorded == 1


def doubled(pred: np.ndarray, dataset: Dataset, rng: Range, config) -> np.ndarray:
    return 2 * pred


def test_a_postprocess_reaching_another_distribution_shares_the_fits(
    ledger, evaluation
):
    first = _run(ledger, evaluation, synthetic.ridge(1))
    post = dataclasses.replace(
        synthetic.ridge(1), postprocess=doubled, postprocess_config=0, name="x2"
    )
    report = _run(ledger, evaluation, post)
    assert report.fits_computed == 0 and report.fits_reused == first.fits_computed


@dataclasses.dataclass(frozen=True)
class TwinLabelSplit:
    def folds(self, dataset):
        return [splits.Fold("twin", ((0, c),), (c, c + 500)) for c in (2000, 3000)]


def test_folds_sharing_a_label_fit_apart_and_a_split_refuses_them(ledger, dataset):
    evaluation = synthetic.evaluation(dataset, split=TwinLabelSplit())
    report = _run(ledger, evaluation, synthetic.ridge(1))
    assert report.fits_computed == 2 and not report.failed
    with pytest.raises(Refused, match=r"fold labels \['twin'\] twice"):
        splits._at_least(TwinLabelSplit().folds(None), 1)


def test_the_modules_lab_run_imports_join_every_id(ledger, evaluation, scratch):
    code = scratch / "exp_j.py"
    code.write_text("SEED = 1\n")
    module = cli._load(str(code))
    first = _run(ledger, evaluation, synthetic.ridge(1), experiments=[module])
    assert _run(ledger, evaluation, synthetic.ridge(1), experiments=[module]).run == ""
    code.write_text("SEED = 2\n")
    again = _run(ledger, evaluation, synthetic.ridge(1), experiments=[module])
    assert again.fits_computed == first.fits_computed and again.scores_recorded == 1
    keys = ledger.events(Event.FIT)[-1]["payload"]["code_keys"]
    assert any(path.endswith("/exp_j.py") for path in keys)


def test_a_sealed_tail_is_validated_by_a_sealed_evaluation_once(ledger):
    rows = synthetic.generate(
        start="2025-01-01", months=12, rows_per_day=20, seed=7, drift_at="2025-07-01"
    )
    dataset = record(
        ledger,
        rows,
        source="sealed",
        params={},
        filters=(),
        targets=(synthetic.TARGET,),
        sealed_from="2025-11-01",
    )
    tail = load(ledger, dataset).index_of("2025-11-01")
    with pytest.raises(
        Refused,
        match=rf"not sealed and fold 2025-11-01 validates rows {tail} to \d+, past "
        rf"row {tail}",
    ):
        _run(ledger, synthetic.evaluation(dataset), synthetic.ridge(1))
    month = splits.CalendarWalkForward("2025-05-01", embargo_timestamps=1)
    ended = dataclasses.replace(month, end="2025-11-01")
    validation = synthetic.evaluation(dataset, split=ended)
    assert _run(ledger, validation, synthetic.ridge(1)).scores_recorded == 1
    after = dataclasses.replace(month, first_cutoff="2025-11-01", min_folds=2)
    test = dataclasses.replace(synthetic.evaluation(dataset, split=after), sealed=True)
    assert _run(ledger, test, synthetic.ridge(1)).scores_recorded == 1
    assert _run(ledger, test, synthetic.ridge(1)).scores_reused == 1
    scorer = dataclasses.replace(test.scorer, directions={"pnl": "min"})
    again = dataclasses.replace(test, scorer=scorer)
    for dry in (True, False):
        error = _run(ledger, again, synthetic.ridge(1), dry=dry).failed["ridge_1m"]
        assert re.search(
            rf"sealed evaluation {test.id} already scored pipeline ridge_1m as score "
            r"\w+; a sealed evaluation scores each pipeline once",
            error,
        )
    unsealed = synthetic.evaluation(synthetic.dataset(ledger))
    with pytest.raises(Refused, match="is sealed and dataset .* has no sealed tail"):
        _run(ledger, dataclasses.replace(unsealed, sealed=True), synthetic.ridge(1))


def test_a_fit_carries_code_identity_and_its_model_reloads_without_pickle(
    ledger, evaluation
):
    pipeline = synthetic.ridge(3)
    _run(ledger, evaluation, pipeline)
    fit = ledger.events(Event.FIT)[0]
    payload = fit["payload"]
    assert "tests/synthetic.py" in payload["import_shas"]
    assert payload["model"]["format"] == "arrow-arrays"
    lock = dict(
        line.split("==", 1)
        for line in ledger.get_blob(payload["env_lock"]["sha"]).decode().split()
    )
    assert {"python", "numpy", "polars"} <= set(lock) and "pytest" not in lock
    assert lock["ml-lab"].count("+") == 1
    run = ledger.latest(Event.RUN, payload["run"])
    assert "commit" in run["payload"]["git"]
    model = pipeline.load(ledger.get_blob(payload["model"]["sha"]))
    assert model.weights.shape == (3,)


def test_a_changed_source_file_is_a_new_fit_and_score_but_the_same_pipeline(
    ledger, tmp_path
):
    code = tmp_path / "steps_v.py"
    code.write_text(
        (REPO / "tests" / "synthetic.py").read_text().replace("SYN", "SYNV")
    )
    module = cli._load(str(code))
    dataset = module.dataset(ledger)
    evaluation = module.evaluation(dataset)
    pipeline = module.ridge(3)
    runs.run(ledger, [pipeline], evaluation, code_root=tmp_path)
    first_score = _latest(ledger, evaluation)["ridge_3m"]
    code.write_text(code.read_text().replace("0.01 * np.sum", "0.02 * np.sum"))
    module = cli._load(str(code))
    report = runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    assert module.ridge(3).id == pipeline.id
    assert report.fits_computed > 0 and report.scores_recorded == 1
    assert _latest(ledger, evaluation)["ridge_3m"] != first_score
    assert ledger.latest(Event.SCORE, first_score) is not None


def test_a_changed_scorer_rescores_without_refitting(ledger, tmp_path):
    steps = tmp_path / "steps_w.py"
    steps.write_text(
        (REPO / "tests" / "synthetic.py").read_text().replace("SYN", "SYNW")
    )
    scoring = tmp_path / "scoring_w.py"
    scoring.write_text(
        "import numpy as np\n"
        "from ml_lab.experiment import Scorer\n"
        "from steps_w import TARGET, sim_metrics\n\n"
        "def sim(pred, dataset, rng, config):\n"
        "    truth = dataset.column(TARGET, rng)\n"
        "    flips = np.abs(np.diff(np.sign(pred), prepend=0.0))\n"
        "    return np.column_stack([np.sign(pred) * truth - 0.001 * flips, flips])\n"
        'scorer = Scorer(sim, sim_metrics, {"pnl": "max", "turnover": "min"})\n'
    )
    module = cli._load(str(steps))
    evaluation = dataclasses.replace(
        module.evaluation(module.dataset(ledger)),
        scorer=cli._load(str(scoring)).scorer,
    )
    runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    first = _latest(ledger, evaluation)["ridge_3m"]
    scoring.write_text(scoring.read_text().replace("0.001 * flips", "0.002 * flips"))
    changed = dataclasses.replace(evaluation, scorer=cli._load(str(scoring)).scorer)
    report = runs.run(ledger, [module.ridge(3)], changed, code_root=tmp_path)
    assert report.fits_computed == 0 and report.scores_recorded == 1
    assert _latest(ledger, evaluation)["ridge_3m"] != first


def test_a_failing_pipeline_is_recorded_and_the_rest_continue_and_a_rerun_resumes(
    ledger, dataset, evaluation
):
    folds = evaluation.split.folds(load(ledger, dataset))
    synthetic.FLAKY_CALLS.clear()
    flaky = synthetic.flaky()
    report = _run(ledger, evaluation, flaky, synthetic.ridge(1))
    assert list(report.failed) == ["flaky"] and "boom" in report.failed["flaky"]
    assert report.fits_computed == 2 + len(folds) and report.scores_recorded == 1
    failure = ledger.sql("SELECT pipeline, error FROM raw_failure")[0]
    assert failure["pipeline"] == flaky.id and "boom" in failure["error"]
    event = ledger.events(Event.FAILED, key=flaky.id)[-1]
    assert (
        "RuntimeError" in ledger.get_blob(event["payload"]["traceback"]["sha"]).decode()
    )
    again = _run(ledger, evaluation, flaky)
    assert again.fits_computed == len(folds) - 2 and again.scores_recorded == 1
    assert set(_latest(ledger, evaluation)) == {"flaky", "ridge_1m"}
    assert _count(ledger, "raw_failure") == 0


def test_a_fit_that_imports_a_distribution_lazily_is_refused():
    pipeline = synthetic.ridge(3)
    dists = identity.imported_dists(pipeline.functions, REPO)
    before_pytest = {name for name in sys.modules if not name.startswith("pytest")}
    with pytest.raises(Refused, match="distributions \\[.*'pytest'"):
        runs._refuse_lazy_imports(pipeline, REPO, dists, before_pytest, "fit")


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_a_step_that_imports_a_repo_module_lazily_is_refused(
    ledger, evaluation, scratch, stage
):
    module, helper = _lazy_steps(scratch, stage)
    lazy = getattr(module, f"lazy_{stage}")
    p = dataclasses.replace(synthetic.ridge(1), **{stage: lazy}, name="lazy")
    report = _run(ledger, evaluation, p)
    assert f"{stage} imported lazily" in report.failed["lazy"]
    assert helper in report.failed["lazy"]
    assert _count(ledger, "raw_prediction") == 0


def test_a_lazy_import_of_an_installed_submodule_is_not_a_repo_module(
    ledger, evaluation, scratch
):
    code = scratch / "lazy_venv_steps.py"
    code.write_text(LAZY_STEPS.format(helper="numpy.ma"))
    module = cli._load(str(code))
    p = dataclasses.replace(synthetic.ridge(1), predict=module.lazy_predict, name="pt")
    report = _run(ledger, evaluation, p)
    assert not report.failed and "pt" in _latest(ledger, evaluation)


def test_the_run_records_the_resolution_file_when_present(ledger, evaluation):
    _run(ledger, evaluation, synthetic.ridge(3))
    run = ledger.events(Event.RUN)[0]["payload"]
    assert run["resolution"] is None or run["resolution"]["path"].startswith(
        ("uv.lock", "requirements")
    )


# Provenance ===========================================================================


def test_untracked_step_modules_are_in_the_run_diff(ledger, tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "new_steps.py").write_text("y = 2\n")
    git = runs._git(ledger, repo)
    diff = ledger.get_blob(git["diff"]["sha"]).decode()
    assert git["dirty"] and "new_steps.py" in diff and "+y = 2" in diff


def test_a_large_untracked_file_is_named_in_the_run_diff_not_diffed(ledger, tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "big.csv").write_bytes(b"0" * (runs.UNTRACKED_DIFF_CAP + 1))
    (repo / "b.py").write_text("y = 2\n")
    diff = ledger.get_blob(runs._git(ledger, repo)["diff"]["sha"]).decode()
    assert "+y = 2" in diff and "not diffed: big.csv" in diff
    assert "+000" not in diff
    assert not (repo / ".git" / "index").read_bytes().count(b"b.py")


def test_git_keeps_the_raw_bytes_of_a_file_that_is_not_utf8(ledger, tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "a.py").write_bytes(b"x = '\xe9t\xe9'\n")
    git = runs._git(ledger, repo)
    assert git["dirty"] and b"+x = '\xe9t\xe9'" in ledger.get_blob(git["diff"]["sha"])


def test_git_survives_a_dangling_untracked_symlink(ledger, tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "link.py").symlink_to(repo / "gone.py")
    (repo / "b.py").write_text("y = 2\n")
    git = runs._git(ledger, repo)
    assert b"+y = 2" in ledger.get_blob(git["diff"]["sha"])


def test_git_before_the_first_commit_records_the_untracked_code(ledger, tmp_path):
    git = runs._git(ledger, _repo(tmp_path / "repo", commit=False))
    assert git["commit"] is None and git["dirty"] is True
    assert b"+x = 1" in ledger.get_blob(git["diff"]["sha"])


def test_a_failing_git_leaves_no_fit_under_an_unrecorded_run(
    ledger, evaluation, monkeypatch
):
    def broken(ledger, root):
        raise RuntimeError("git broke")

    monkeypatch.setattr(runs, "_git", broken)
    with pytest.raises(RuntimeError, match="git broke"):
        _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(3))
    assert _count(ledger, "raw_run") == _count(ledger, "raw_fit") == 0
    assert _count(ledger, "raw_failure") == 0


# Command line =========================================================================


def test_fy_run_merges_pipelines_from_several_modules_and_defaults_the_dataset(
    ledger, dataset, evaluation, scratch, capsys
):
    extra = scratch / "agent7.py"
    extra.write_text("from tests.synthetic import ridge\npipelines = [ridge(12)]\n")
    by_path = str(REPO / "tests" / "synthetic.py")
    assert cli.main(["--root", str(ledger.root), "run", by_path, str(extra)]) == 0
    out = capsys.readouterr().out
    assert "scores 4" in out and "fit ridge_12m " in out and "score ridge_12m " in out
    assert set(_latest(ledger, evaluation)) == {
        "ridge_1m",
        "ridge_3m",
        "ridge_6m",
        "ridge_12m",
    }
    again = ["--root", str(ledger.root), "run", by_path, str(extra)]
    assert cli.main([*again, "--dataset", dataset[:6]]) == 0
    assert "up to date" in capsys.readouterr().out


def test_fy_run_refuses_bad_experiments(ledger, dataset, tmp_path, capsys):
    extra = tmp_path / "bare.py"
    extra.write_text("from tests.synthetic import ridge\npipelines = [ridge(12)]\n")
    root = ["--root", str(ledger.root)]
    assert cli.main([*root, "run", str(extra)]) == 1
    assert "0 modules declare evaluations" in capsys.readouterr().err
    assert cli.main([*root, "run", "nope.decl"]) == 1
    assert "No module named 'nope'" in capsys.readouterr().err
    assert cli.main([*root, "run", "tests.synthetic", "--dataset", "zzz"]) == 1
    assert "0 matches" in capsys.readouterr().err
    empty = tmp_path / "none.py"
    empty.write_text("from tests.synthetic import evaluations\npipelines = []\n")
    assert cli.main([*root, "run", str(empty)]) == 1
    assert "no pipelines declared" in capsys.readouterr().err


def test_fy_run_refuses_a_shared_ledger_without_dataset_and_takes_a_source(
    ledger, dataset, evaluation, capsys
):
    record(
        ledger,
        synthetic.generate(
            start="2025-01-01", months=2, rows_per_day=5, seed=1, drift_at="2025-02-01"
        ),
        source="other",
        params={},
        filters=(synthetic.keep_all,),
        targets=(synthetic.TARGET,),
    )
    root = ["--root", str(ledger.root)]
    assert cli.main([*root, "run", "tests.synthetic"]) == 1
    assert "several sources" in capsys.readouterr().err
    assert cli.main([*root, "run", "tests.synthetic", "--dataset", "synthetic"]) == 0
    assert set(_latest(ledger, evaluation)) == {"ridge_1m", "ridge_3m", "ridge_6m"}


def test_two_names_on_one_declaration_are_refused(ledger, evaluation):
    with pytest.raises(Refused, match="ridge_1m and twin"):
        _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(1).named("twin"))
    assert Event.PIPELINE not in _types(ledger)


def test_fy_run_refuses_a_root_without_a_ledger_and_creates_none(tmp_path, capsys):
    empty = tmp_path / "empty"
    assert cli.main(["--root", str(empty), "run", "tests.synthetic"]) == 1
    assert f"no ledger at {empty.resolve()}" in capsys.readouterr().err
    assert not empty.exists()
    assert cli.main(["--root", str(empty), "ingest", "tests.synthetic"]) == 0
    run = ["--root", str(empty), "run", "tests.synthetic", "--dataset", "zzz"]
    assert cli.main(run) == 1
    assert f"0 matches in {empty.resolve()}" in capsys.readouterr().err


def test_lab_run_flushes_each_progress_line(ledger, dataset, monkeypatch):
    class Recording(io.StringIO):
        flushed = ""

        def flush(self):
            self.flushed = self.getvalue()

    out = Recording()
    monkeypatch.setattr(sys, "stdout", out)

    def fake(ledger, pipelines, evaluation, *, log, **kwargs):
        log("fit ridge_1m 2025-05-01 0.1s")
        assert out.flushed.endswith("fit ridge_1m 2025-05-01 0.1s\n")
        return runs.RunReport()

    monkeypatch.setattr(runs, "run", fake)
    assert cli.main(["--root", str(ledger.root), "run", "tests.synthetic"]) == 0


def test_a_source_written_by_two_actors_needs_an_id(
    ledger, dataset, evaluation, monkeypatch, capsys
):
    monkeypatch.setenv("ML_LAB_ACTOR", "other-agent")
    second = synthetic.dataset(ledger, seed=8)
    assert second != dataset
    with pytest.raises(Refused, match="several actors") as info:
        cli._dataset(ledger, "synthetic")
    assert dataset[:8] in str(info.value) and "other-agent" in str(info.value)
    with pytest.raises(Refused, match="several actors"):
        cli._dataset(ledger, None)
    assert cli._dataset(ledger, dataset[:8]) == dataset
    assert cli.main(["--root", str(ledger.root), "run", "tests.synthetic"]) == 1
    assert "several actors" in capsys.readouterr().err


def test_lab_run_hashes_from_the_git_root_of_the_evaluations_module(
    ledger, dataset, tmp_path, monkeypatch
):
    seen = {}

    def fake(ledger, pipelines, evaluation, **kwargs):
        seen.update(kwargs)
        return runs.RunReport()

    monkeypatch.setattr(runs, "run", fake)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    agent = tmp_path / "agent_root.py"
    agent.write_text(
        "import dataclasses\nfrom tests import synthetic\n\n"
        "def fit(dataset, train, config):\n"
        "    return synthetic.ridge_fit(dataset, train, config)\n\n"
        "pipelines = [dataclasses.replace(synthetic.ridge(12), fit=fit)]\n"
    )
    run = ["--root", str(ledger.root), "run", str(agent), "tests.synthetic"]
    assert cli.main(run) == 0 and sys.dont_write_bytecode
    assert seen["code_root"] == REPO
    names = [m.__name__ for m in seen["experiments"]]
    assert names == ["agent_root", "tests.synthetic"]


def test_lab_run_skips_a_failed_pipeline_in_later_evaluations(
    ledger, dataset, scratch, capsys
):
    sweep = scratch / "skip.py"
    sweep.write_text(
        "import dataclasses\nfrom tests import synthetic\n"
        "pipelines = [synthetic.ridge(1), dataclasses.replace(synthetic.ridge(3), "
        "predict=synthetic.peeking_predict, name='peek')]\n"
        "def evaluations(d):\n"
        "    return [dataclasses.replace(synthetic.evaluation(d, cost=c), name=f'c{c}')"
        " for c in (0.001, 0.01)]\n"
    )
    assert cli.main(["--root", str(ledger.root), "run", str(sweep)]) == 1
    out = capsys.readouterr().out
    assert out.count("lab run: peek failed") == 1
    assert "failed, skipped in later evaluations: peek" in out
    assert re.search(r"total over 2 evaluations: computed fits \d+", out)
    assert _count(ledger, "raw_failure") == 1 and _count(ledger, "score_latest") == 2


# Read back ============================================================================


def test_the_views_read_with_sqlite_alone(ledger, dataset, evaluation, tmp_path):
    folds = evaluation.split.folds(load(ledger, dataset))
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(6))
    db = sqlite3.connect(tmp_path / "ml-lab" / "ml_lab.sqlite")
    metrics = len(evaluation.directions)
    assert db.execute("SELECT COUNT(*) FROM score_fold").fetchone()[0] == (
        2 * len(folds) * metrics
    )
    assert db.execute("SELECT COUNT(*) FROM score_aggregate").fetchone()[0] == (
        2 * metrics
    )
    assert db.execute("SELECT COUNT(*) FROM score_latest").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM raw_fit").fetchone()[0] == 2 * len(folds)
    assert db.execute("SELECT COUNT(*) FROM score_fit").fetchone()[0] == 2 * len(folds)
    assert (
        db.execute("SELECT SUM(duration_s) FROM score_fit").fetchone()
        == db.execute("SELECT SUM(duration_s) FROM raw_fit").fetchone()
    )
    assert db.execute("SELECT DISTINCT source FROM score_latest").fetchall() == [
        ("synthetic",)
    ]
    scored, folds_, std, sha = db.execute(
        "SELECT score, folds, fold_std, series FROM score_aggregate"
    ).fetchone()
    assert folds_ == len(folds) and std > 0
    stored = ledger.latest(Event.SCORE, scored)["payload"]["series"]
    assert stored["sha"] == sha
    assert len(_load_series(ledger, sha)) == sum(stored["fold_rows"])
    assert db.execute("SELECT resolution, pipelines FROM raw_run").fetchone()[1]


def test_board_reads_names_and_values_without_a_join(ledger, evaluation):
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(6))
    rows = ledger.sql("SELECT * FROM board WHERE metric = 'pnl'")
    joined = ledger.sql(
        "SELECT l.name, a.value FROM score_latest l "
        "JOIN score_aggregate a ON a.score = l.score "
        "WHERE a.metric = 'pnl'"
    )
    assert {r["name"]: r["value"] for r in rows} == {
        r["name"]: r["value"] for r in joined
    }
    assert list(rows[0])[:4] == ["source", "evaluation_name", "name", "metric"]
    assert rows[0]["source"] == "synthetic" and rows[0]["folds"] > 1


def test_head_to_head_is_the_fold_by_fold_difference_and_the_series_is_named(
    ledger, evaluation
):
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(6))
    scores = _latest(ledger, evaluation)
    values = {}
    for name, score in scores.items():
        rows = ledger.sql(
            "SELECT value FROM score_fold WHERE score = ? AND metric = 'pnl' "
            "ORDER BY fold",
            (score,),
        )
        values[name] = np.array([r["value"] for r in rows])
    d = values["ridge_1m"] - values["ridge_6m"]
    row = ledger.sql(
        "SELECT * FROM head_to_head WHERE name = 'ridge_1m' AND reference_name = "
        "'ridge_6m' AND metric = 'pnl'"
    )[0]
    assert row["folds"] == len(d) and row["wins"] == int((d > 0).sum())
    assert np.isclose(row["mean_delta"], d.mean())
    assert np.isclose(row["delta_std"], d.std(ddof=1))
    assert np.isclose(row["t"], d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    sha = ledger.sql("SELECT series FROM score_aggregate LIMIT 1")
    assert list(formats.arrays_load(ledger.get_blob(sha[0]["series"]))) == [
        "pnl",
        "flips",
    ]


def test_head_to_head_names_its_evaluation_and_reconciles_with_the_pooled_value(
    ledger, evaluation
):
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(6))
    row = ledger.sql(
        "SELECT * FROM head_to_head WHERE name = 'ridge_1m' AND reference_name = "
        "'ridge_6m' AND metric = 'pnl'"
    )[0]
    assert (row["evaluation_name"], row["source"]) == ("", "synthetic")
    agg = {
        r["score"]: r
        for r in ledger.sql(
            "SELECT score, value, fold_mean FROM score_aggregate WHERE metric = 'pnl'"
        )
    }
    a, b = agg[row["score"]], agg[row["reference_score"]]
    assert np.isclose(row["pooled_delta"], a["value"] - b["value"])
    assert np.isclose(row["mean_delta"], a["fold_mean"] - b["fold_mean"])


def test_wins_follow_the_direction_recorded_with_each_score(
    ledger, tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    steps = tmp_path / "steps_x.py"
    steps.write_text(
        (REPO / "tests" / "synthetic.py").read_text().replace("SYN", "SYNX")
    )
    scoring = tmp_path / "scoring_x.py"
    scoring.write_text(
        "from ml_lab.experiment import Scorer\n"
        "from steps_x import sign_sim, sim_metrics\n\n"
        "def sim(pred, dataset, rng, config):\n"
        "    return sign_sim(pred, dataset, rng, config)\n"
        'scorer = Scorer(sim, sim_metrics, {"pnl": "max", "turnover": "min"})\n'
    )
    module = cli._load(str(steps))
    evaluation = dataclasses.replace(
        module.evaluation(module.dataset(ledger)),
        scorer=cli._load(str(scoring)).scorer,
    )
    pipelines = [module.ridge(1), module.ridge(3)]
    runs.run(ledger, pipelines, evaluation, code_root=tmp_path)
    query = (
        "SELECT folds, wins FROM head_to_head WHERE name = 'ridge_1m' "
        "AND metric = 'pnl'"
    )
    (before,) = ledger.sql(query)
    declared = evaluation.id
    scoring.write_text(scoring.read_text().replace('"pnl": "max"', '"pnl": "min"'))
    scorer = importlib.reload(cli._load(str(scoring))).scorer
    flipped = dataclasses.replace(evaluation, scorer=scorer)
    assert flipped.id == declared
    runs.run(ledger, pipelines, flipped, code_root=tmp_path)
    (after,) = ledger.sql(query)
    assert 0 < before["wins"] < before["folds"]
    assert after["wins"] == before["folds"] - before["wins"]


def test_a_one_fold_holdout_pairs_the_stored_series_by_row(ledger, dataset):
    holdout = synthetic.evaluation(dataset, split=splits.Holdout(0.7))
    _run(ledger, holdout, synthetic.ridge(1), synthetic.ridge(6))
    row = ledger.sql(
        "SELECT * FROM head_to_head WHERE name = 'ridge_1m' AND metric = 'pnl'"
    )[0]
    assert row["folds"] == 1 and row["delta_std"] is None and row["t"] is None
    series = ledger.sql(
        "SELECT score, series, fold_rows FROM score_aggregate "
        "WHERE metric = 'pnl' AND score IN (?, ?)",
        (row["score"], row["reference_score"]),
    )
    assert series[0]["fold_rows"] == series[1]["fold_rows"]
    pnl = {
        r["score"]: formats.arrays_load(ledger.get_blob(r["series"]))["pnl"]
        for r in series
    }
    d = pnl[row["score"]] - pnl[row["reference_score"]]
    assert np.isclose(d.sum(), row["pooled_delta"]) and d.std(ddof=1) > 0


def test_the_log_reads_as_it_stood(ledger, evaluation):
    _run(ledger, evaluation, synthetic.ridge(6))
    before = ledger.events()[-1]["seq"]
    _run(ledger, evaluation, synthetic.ridge(1))
    assert _count(ledger, "raw_score WHERE seq <= ?", before) == 1
    assert len(ledger.events(Event.SCORE)) == 2


def test_connecting_drops_views_a_previous_build_left_behind(ledger):
    ledger._db.execute("CREATE VIEW paired_score AS SELECT * FROM head_to_head")
    names = Ledger(ledger.root).sql("SELECT name FROM sqlite_master WHERE type='view'")
    assert "paired_score" not in {r["name"] for r in names}


def _nan_metrics(series):
    return {"ic": float("nan")}


def nan_score(pred, dataset, rng, config):
    return np.zeros(rng[1] - rng[0])


nan_scorer = Scorer(nan_score, _nan_metrics, {"ic": "max"})


def test_a_non_finite_metric_fails_the_pipeline_naming_the_fold(ledger, evaluation):
    nan_evaluation = dataclasses.replace(evaluation, scorer=nan_scorer, config=None)
    error = _run(ledger, nan_evaluation, synthetic.ridge(3)).failed["ridge_3m"]
    assert "metrics ['ic'] are not finite on fold 2025-05-01" in error
    assert not ledger.events(Event.SCORE) and _count(ledger, "raw_failure") == 1


@dataclasses.dataclass(frozen=True)
class DottedSplit:
    def folds(self, dataset):
        cuts = (2000, 3000, 4000)
        return [
            splits.Fold(f"f{k}", ((0, c),), (c, c + 500)) for k, c in enumerate(cuts)
        ]


def _dotted_metrics(series):
    return {"hit.rate": float((series > 0).mean())}


def dotted_score(pred, dataset, rng, config):
    return np.sign(pred) * dataset.column(synthetic.TARGET, rng)


dotted_scorer = Scorer(dotted_score, _dotted_metrics, {"hit.rate": "max"})


def test_a_dotted_metric_keeps_its_series_and_counts_wins(ledger, dataset):
    evaluation = Evaluation(dataset=dataset, split=DottedSplit(), scorer=dotted_scorer)
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(3))
    rows = ledger.sql("SELECT series, fold_rows FROM score_aggregate")
    assert rows and all(r["series"] and r["fold_rows"] for r in rows)
    rows = ledger.sql("SELECT wins FROM head_to_head")
    assert rows and all(r["wins"] is not None for r in rows)


def _two_scores(ledger: Ledger, values: list[float]):
    ledger.append(Event.DATASET, "d", "d", {"source": "s"}, id="d")
    ledger.append(Event.EVALUATION, "e", "e", {"dataset": "d", "metrics": {}})
    for p in ("a", "b"):
        ledger.append(Event.PIPELINE, p, p, {"name": p})
    _score(ledger, "sa", "a", values)
    _score(ledger, "sb", "b", [0.0] * len(values))


def test_fold_and_delta_std_survive_a_large_metric_offset(ledger):
    _two_scores(ledger, [1e9 + i * i for i in range(4)])
    std = pytest.approx(np.std([0, 1, 4, 9], ddof=1), rel=1e-6)
    (row,) = ledger.sql("SELECT fold_std FROM score_aggregate WHERE score = 'sa'")
    assert row["fold_std"] == std
    (row,) = ledger.sql("SELECT delta_std FROM head_to_head WHERE pipeline = 'a'")
    assert row["delta_std"] == std


def test_a_null_fold_metric_is_left_out_of_the_fold_count_and_spread(ledger):
    _two_scores(ledger, [1.0, float("nan"), 4.0, 9.0])
    std = pytest.approx(np.std([1.0, 4.0, 9.0], ddof=1))
    query = "SELECT folds, fold_std FROM score_aggregate WHERE score = 'sa'"
    (row,) = ledger.sql(query)
    assert (row["folds"], row["fold_std"]) == (3, std)
    (row,) = ledger.sql(
        "SELECT folds, delta_std FROM head_to_head WHERE pipeline = 'a'"
    )
    assert (row["folds"], row["delta_std"]) == (3, std)


def _auc(series: np.ndarray) -> dict[str, float]:
    p, y = series[:, 0], series[:, 1] == 1
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    n1, n0 = y.sum(), (~y).sum()
    return {"auc": float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))}


def auc_series(pred: np.ndarray, dataset: Dataset, rng: Range, config: None):
    return np.column_stack([pred, dataset.column("y", rng)])


auc_score = Scorer(auc_series, _auc, {"auc": "max"}, ("p", "y"))


def prior_fit(dataset: Dataset, train: Segments, config: None) -> float:
    return float(np.mean(np.concatenate([dataset.column("y", s) for s in train])))


def prior_predict(model: float, dataset: Dataset, rng: Range) -> np.ndarray:
    return np.full(rng[1] - rng[0], model)


def prior_save(model: float) -> bytes:
    return formats.arrays_save({"p": np.array([model])})


def prior_load(payload: bytes) -> float:
    return float(formats.arrays_load(payload)["p"][0])


def test_a_constant_per_fold_scores_auc_one_half_by_fold_mean_not_pooled(ledger):
    rate = np.repeat([0.1, 0.3, 0.5, 0.7], 100)
    y = (np.random.default_rng(1).random(400) < rate).astype(float)
    rows = Dataset({"x": np.zeros(400), "y": y})
    dataset = record(ledger, rows, source="auc", params={}, filters=(), targets=("y",))
    evaluation = Evaluation(
        dataset, splits.WalkForward(100, 100, 100, min_folds=3), auc_score
    )
    prior = Pipeline(
        prior_fit, prior_predict, prior_save, prior_load, None, "arrow-arrays", name="p"
    )
    assert not _run(ledger, evaluation, prior).failed
    row = ledger.sql("SELECT * FROM board")[0]
    assert row["folds"] == 3 and row["fold_mean"] == 0.5 and row["value"] != 0.5
    assert list(row).index("fold_mean") < list(row).index("value")


# Storage ==============================================================================


def test_concurrent_connects_rebuild_the_views_whole(tmp_path):
    failures = []

    def connect():
        try:
            Ledger(tmp_path).sql("SELECT COUNT(*) FROM head_to_head")
        except Exception as e:
            failures.append(e)

    workers = [threading.Thread(target=connect) for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert failures == []


def test_ulids_sort_and_do_not_collide():
    ids = [identity.ulid() for _ in range(1000)]
    assert (
        all(len(i) == 26 for i in ids) and len(set(ids)) == 1000 and ids == sorted(ids)
    )


def test_blobs_are_written_atomically_under_their_sha(ledger):
    sha = ledger.put_blob(b"hello")
    assert (ledger.blobs / sha).exists() and ledger.get_blob(sha) == b"hello"
    assert sha == identity.bytes_hash(b"hello")
    assert not list(ledger.blobs.glob(".tmp-*"))


def test_git_blob_sha_matches_git():
    assert (
        identity.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    )


def test_a_zip_member_named_like_the_manifest_is_refused():
    with pytest.raises(ValueError, match="manifest"):
        formats.zip_save({formats.MANIFEST: (formats.Format.TEXT, b"{}")})


def test_arrays_round_trip_when_every_array_is_empty():
    out = formats.arrays_load(formats.arrays_save({"e": np.zeros((0, 3))}))
    assert out["e"].shape == (0, 3)


def test_a_seconds_datetime_column_is_stored_in_nanoseconds():
    column = np.datetime64("2025-01-01", "s") + np.arange(48).astype("timedelta64[s]")
    loaded = rows_load(rows_save(_dataset(z=column)))
    np.testing.assert_array_equal(loaded.columns["z"], column.astype("datetime64[ns]"))


# Panel ================================================================================


def test_a_panel_grids_a_dataset_by_time_and_key_and_back():
    ts = np.array(["2025-01-01"] * 2 + ["2025-01-02"], dtype="datetime64[s]")
    dataset = Dataset(
        {"stock": np.array([1, 2, 1]), "px": np.array([10.0, 20.0, 11.0])}, ts
    )
    panel = Panel(dataset, "stock")
    grid = panel.grid("px")
    assert panel.shape == (2, 2) and np.isnan(grid[1, 1])
    assert np.array_equal(grid[:, 0], [10.0, 11.0])
    assert np.array_equal(panel.rows(grid), dataset.columns["px"])
    assert panel.grid("px", dtype=np.float32).dtype == np.float32
    demeaned = grid - np.nanmean(grid, axis=1, keepdims=True)
    assert np.allclose(panel.rows(demeaned), [-5.0, 5.0, 0.0])
    assert Panel(dataset.upto(2), "stock").shape == (1, 2)
    assert Panel(dataset.upto(2), "stock", keys=[1, 2, 3]).shape == (1, 3)
    with pytest.raises(ValueError, match="is not in keys"):
        Panel(dataset, "stock", keys=[1])
    with pytest.raises(ValueError, match="timestamps"):
        Panel(Dataset({"stock": np.array([1])}), "stock")


def test_a_refused_predict_records_nothing_and_every_rerun_refuses(ledger, dataset):
    peek = dataclasses.replace(synthetic.ridge(1), predict=synthetic.peeking_predict)
    for cost in (0.001, 0.002, 0.003):
        report = _run(ledger, synthetic.evaluation(dataset, cost=cost), peek)
        assert "predict read past row" in report.failed["ridge_1m"]
    assert Event.FAILED in _types(ledger)
    assert not ledger.events(Event.PREDICTIONS) and not ledger.events(Event.SCORE)


def test_a_postprocess_is_probed_with_its_predictions_cut(ledger, evaluation):
    centred = dataclasses.replace(
        synthetic.ridge(1), postprocess=synthetic.centre_window, name="centred"
    )
    report = _run(ledger, evaluation, centred)
    assert "postprocess read past row" in report.failed["centred"]


def test_a_save_that_drops_state_is_refused_before_its_fit_is_recorded(
    ledger, evaluation
):
    lossy = dataclasses.replace(synthetic.ridge(1), save=synthetic.biasless_save)
    report = _run(ledger, evaluation, lossy)
    assert "loaded from its saved bytes" in report.failed["ridge_1m"]
    assert not ledger.events(Event.FIT)


def test_rows_sharing_a_timestamp_are_one_instant_to_the_probe(ledger):
    split = splits.WalkForward(200, 50, 50, min_folds=2)
    evaluation = synthetic.evaluation(synthetic.panel(ledger), split=split)
    xs = dataclasses.replace(
        synthetic.ridge(0), predict=synthetic.demean_at_timestamp, name="xs"
    )
    peek = dataclasses.replace(synthetic.ridge(0), predict=synthetic.peeking_predict)
    report = _run(ledger, evaluation, xs, peek)
    assert set(report.failed) == {"ridge_0m"} and report.scores_recorded == 1


def test_a_revealed_target_is_a_feature_no_earlier_than_its_lag(ledger):
    split = splits.WalkForward(200, 50, 50, embargo_rows=2, min_folds=2)
    two = datetime.timedelta(minutes=2)
    dataset = synthetic.panel(ledger, reveal={synthetic.TARGET: two})
    assert dataset != synthetic.panel(ledger)
    evaluation = synthetic.evaluation(dataset, split=split)
    base = synthetic.ridge(0)
    known = synthetic.featured(base, synthetic.label_two_minutes_back, "known")
    early = synthetic.featured(base, synthetic.label_one_minute_back, "early")
    report = _run(ledger, evaluation, known, early)
    assert set(report.failed) == {"early"} and "f0_lag" in report.failed["early"]
    unrevealed = synthetic.evaluation(synthetic.panel(ledger), split=split)
    assert synthetic.TARGET in _run(ledger, unrevealed, known).failed["known"]


def test_the_features_probe_catches_a_day_lookahead(ledger, dataset):
    no_embargo = splits.CalendarWalkForward(first_cutoff="2025-05-01")
    evaluation = synthetic.evaluation(dataset, split=no_embargo)
    day = synthetic.featured(synthetic.ridge(1), synthetic.day_mean, "day_mean")
    report = _run(ledger, evaluation, day)
    error = report.failed["day_mean"]
    assert "features read past row" in error and ": f0_lag at row" in error
    assert "of the largest |f0_lag|" in error
