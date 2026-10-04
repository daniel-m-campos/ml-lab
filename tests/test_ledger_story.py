"""The ledger story on synthetic data: ingest, declare, run, read back with SQL.

Mirrors docs/user-stories.md; every event type in docs/spec.md is written and read back.
"""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import sqlite3
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from ml_lab import cli, formats, identity, runs, splits
from ml_lab.dataset import data_id, load, record
from ml_lab.ledger import Event, Ledger, Refused
from ml_lab.panel import Panel
from ml_lab.runs import _load_predictions
from ml_lab.session import Session
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


def _run(ledger, evaluation, *pipelines):
    return runs.run(ledger, list(pipelines), evaluation, code_root=REPO)


def _types(ledger) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in ledger.events():
        out[e["type"]] = out.get(e["type"], 0) + 1
    return out


def _load_series(ledger, sha: str) -> np.ndarray:
    return np.column_stack(list(formats.arrays_load(ledger.get_blob(sha)).values()))


def _latest(ledger, evaluation) -> dict[str, str]:
    rows = ledger.sql(
        "SELECT p.name, l.score FROM latest_score l "
        "JOIN pipeline p ON p.id = l.pipeline "
        "WHERE l.evaluation = ?",
        (evaluation.id,),
    )
    return {r["name"]: r["score"] for r in rows}


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
    assert ledger.sql("SELECT source FROM dataset")[0] == {"source": "synthetic"}


# Declarations =========================================================================


def test_a_pipeline_name_is_a_label_and_a_config_change_is_a_new_id():
    a = synthetic.ridge(3)
    assert a.named("other").id == a.id
    assert a.with_config(alpha=2.0).id != a.id


def test_any_evaluation_field_change_is_a_new_evaluation_id(dataset, evaluation):
    cost = synthetic.evaluation(dataset, cost=0.01)
    one = dataclasses.replace(
        evaluation, split=dataclasses.replace(evaluation.split, horizons=(1,))
    )
    assert len({evaluation.id, cost.id, one.id}) == 3


def test_the_schedule_is_embargoed_and_stored_once_in_the_evaluation_event(
    ledger, dataset, evaluation
):
    session = load(ledger, dataset)
    folds = evaluation.split.folds(session)
    assert len(folds) >= evaluation.split.min_folds
    for fold in folds:
        skipped = session.ts[fold.train[-1][1] : fold.windows["1"][0]]
        assert len(np.unique(skipped)) == evaluation.split.embargo_timestamps
    _run(ledger, evaluation, synthetic.ridge(3))
    stored = ledger.latest(Event.EVALUATION, evaluation.id)["payload"]
    assert (
        len(stored["folds"]) == len(folds)
        and stored["metrics"] == evaluation.directions
    )


def test_row_splits_make_disjoint_embargoed_folds(ledger, dataset):
    session = load(ledger, dataset)
    n = session.rows
    folds = splits.BlockedKFold(k=4, embargo_rows=10).folds(session)
    assert [f.label for f in folds] == [f"block {i}" for i in range(4)]
    assert folds[0].train == ((folds[0].windows["test"][1] + 10, n),)
    middle = folds[1]
    lo, hi = middle.windows["test"]
    assert middle.train == ((0, lo - 10), (hi + 10, n))
    assert sum(hi - lo for f in folds for lo, hi in f.windows.values()) == n
    (hold,) = splits.Holdout(train_fraction=0.8, embargo_rows=5).folds(session)
    assert (
        hold.train == ((0, round(0.8 * n)),)
        and hold.windows["test"][0] == round(0.8 * n) + 5
    )


def test_a_clockless_session_fits_on_segments_and_refuses_walk_forward(ledger):
    bare = Session(
        {
            "f0": np.arange(100.0),
            "f1": np.ones(100),
            "f2": np.zeros(100),
            "ret_1": np.arange(100.0),
        }
    )
    assert bare.matrix(((0, 10), (90, 100)), ("f0", "f1")).shape == (20, 2)
    with pytest.raises(ValueError, match="no timestamps"):
        splits.CalendarWalkForward(first_cutoff="2025-01-01").folds(bare)
    assert synthetic.ridge_fit(
        bare, ((0, 50), (60, 100)), synthetic.RidgeConfig(0, 1.0)
    ).weights.shape == (3,)


def test_row_walk_forward_steps_by_rows_and_names_windows_only_when_asked(
    ledger, dataset
):
    session = load(ledger, dataset)
    plain = splits.WalkForward(
        first_cutoff_rows=1000, step_rows=500, window_rows=500, embargo_rows=10
    )
    folds = plain.folds(session)
    assert len(folds) == (session.rows - 1000) // 500
    assert folds[0].train == ((0, 990),) and folds[0].windows == {"test": (1000, 1500)}
    assert folds[1].label == "row 1500"
    named = dataclasses.replace(plain, horizons=(1, 2)).folds(session)
    assert named[0].windows == {"1": (1000, 1500), "2": (1500, 2000)}
    evaluation = synthetic.evaluation(dataset, split=plain)
    report = _run(ledger, evaluation, synthetic.ridge(0))
    assert report.fits_computed == len(folds) and report.scores_recorded == 1


def test_a_blocked_kfold_evaluation_scores_one_test_window_per_fold(ledger, dataset):
    evaluation = synthetic.evaluation(
        dataset, split=splits.BlockedKFold(k=3, embargo_rows=20)
    )
    report = _run(ledger, evaluation, synthetic.ridge(0))
    assert report.fits_computed == 3 and report.scores_recorded == 1
    rows = ledger.sql(
        "SELECT DISTINCT window FROM fold_score WHERE evaluation = ?", (evaluation.id,)
    )
    assert [r["window"] for r in rows] == ["test"]
    fit = ledger.sql("SELECT train, label FROM fit ORDER BY seq LIMIT 2")[1]
    assert fit["label"] == "block 1" and len(json.loads(fit["train"])) == 2


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
    session = Session({"x": np.arange(12.0)}, (days[:, None] + hours).ravel())
    folds = splits.CalendarWalkForward(
        "2019-06-08", unit="day", horizons=(1,), min_folds=1
    ).folds(session)
    assert [f.label for f in folds] == [f"2019-06-1{d}" for d in range(4)]
    assert folds[0].windows == {"1": (2, 4)} and folds[0].train == ((0, 2),)
    assert folds[-1].windows == {"1": (8, 10)}
    with pytest.raises(Refused, match="plus 1 months reaches"):
        splits.CalendarWalkForward("2019-06-12", horizons=(1,)).folds(session)


def test_a_calendar_split_stops_at_end_and_embargoes_whole_timestamps():
    d = np.arange(np.datetime64("2021-01-04"), np.datetime64("2022-06-25"))
    d = d[np.is_busday(d)]
    session = Session({"x": np.arange(d.size, dtype=float)}, d)
    folds = splits.CalendarWalkForward(
        "2021-03-01",
        every=2,
        window=2,
        horizons=(1,),
        embargo_timestamps=2,
        end="2021-12-06",
        min_folds=1,
    ).folds(session)
    last = max(session.ts[hi - 1] for f in folds for _, hi in f.windows.values())
    assert last < np.datetime64("2021-12-06")
    gaps = [
        len(np.unique(session.ts[f.train[0][1] : f.windows["1"][0]])) for f in folds
    ]
    assert gaps == [2] * len(folds) and len(folds) == 4


def test_the_clock_keeps_nanoseconds_and_the_dataset_id_sees_them():
    t0 = np.datetime64("2019-06-10T09:00:00", "ns")
    a = Session({"x": np.zeros(2)}, t0 + np.array([1, 2], "timedelta64[ns]"))
    b = Session({"x": np.zeros(2)}, t0 + np.array([1, 3], "timedelta64[ns]"))
    assert data_id(a) != data_id(b)
    assert np.array_equal(formats.session_load(formats.session_save(a)).ts, a.ts)


# Run ==================================================================================


def test_a_run_posts_fits_predictions_and_one_score_and_a_rerun_writes_nothing(
    ledger, dataset, evaluation
):
    folds = evaluation.split.folds(load(ledger, dataset))
    first = _run(ledger, evaluation, synthetic.ridge(3))
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds) * len(evaluation.split.horizons)
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
        len(folds) * len(evaluation.split.horizons),
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


def test_targets_are_hidden_inside_the_window(ledger, dataset, evaluation):
    cheat = dataclasses.replace(synthetic.ridge(1), predict=synthetic.cheating_predict)
    _run(ledger, evaluation, cheat)
    pred = _load_predictions(ledger, ledger.events(Event.PREDICTIONS)[0]["id"])
    assert np.isnan(pred).all()


def test_a_predict_that_reads_the_future_inside_its_window_is_refused(
    ledger, dataset, evaluation
):
    peek = dataclasses.replace(synthetic.ridge(1), predict=synthetic.peeking_predict)
    report = _run(ledger, evaluation, peek)
    assert "reads rows after the one it predicts" in report.failed["ridge_1m"]
    assert ledger.events(Event.SCORE) == []


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


def test_untracked_step_modules_are_in_the_run_diff(ledger, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", repo, "add", "a.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "a",
        ],
        check=True,
    )
    (repo / "new_steps.py").write_text("y = 2\n")
    git = runs._git(ledger, repo)
    diff = ledger.get_blob(git["diff"]["sha"]).decode()
    assert git["dirty"] and "new_steps.py" in diff and "+y = 2" in diff


def test_a_postprocess_shares_the_fit_and_is_its_own_prediction(
    ledger, dataset, evaluation
):
    folds = evaluation.split.folds(load(ledger, dataset))
    windows = sum(len(f.windows) for f in folds)
    plain = synthetic.ridge(1)
    doubled = dataclasses.replace(
        plain, postprocess=synthetic.scale.configured(factor=2.0), name="ridge_1m_x2"
    )
    assert doubled.id != plain.id
    report = _run(ledger, evaluation, plain, doubled)
    assert report.fits_computed == len(folds) and report.fits_reused == len(folds)
    assert report.predictions_computed == 2 * windows
    assert report.predictions_reused == 0 and report.scores_recorded == 2
    post = [e for e in ledger.events(Event.PREDICTIONS) if "raw" in e["payload"]]
    assert len(post) == windows
    assert post[0]["payload"]["postprocess"]["kwargs"] == {"factor": 2.0}
    raw = _load_predictions(ledger, post[0]["payload"]["raw"])
    assert np.allclose(_load_predictions(ledger, post[0]["id"]), 2 * raw)
    again = _run(ledger, evaluation, plain, doubled)
    assert again.run == "" and again.predictions_reused == 2 * windows


def test_a_blend_reuses_its_members_fits_and_predictions(ledger, dataset, evaluation):
    folds = evaluation.split.folds(load(ledger, dataset))
    windows = sum(len(f.windows) for f in folds)
    one, three = synthetic.ridge(1), synthetic.ridge(3)
    _run(ledger, evaluation, one, three)
    report = _run(ledger, evaluation, synthetic.blend(one, three))
    assert report.fits_reused == 2 * len(folds) and report.fits_computed == len(folds)
    assert report.predictions_reused == 2 * windows
    assert report.predictions_computed == windows + 2 * len(folds)
    assert report.scores_recorded == 1
    blended = [e for e in ledger.events(Event.PREDICTIONS) if "members" in e["payload"]]
    first = next(e for e in blended if e["payload"]["window"] != "train:0")
    parts = [_load_predictions(ledger, p) for p in first["payload"]["members"]]
    assert np.allclose(_load_predictions(ledger, first["id"]), np.mean(parts, axis=0))
    fit = ledger.latest(Event.FIT, first["payload"]["fit"])
    assert len(fit["payload"]["members"]) == 2
    stands_on = ledger.sql(
        "SELECT COUNT(*) AS n FROM score_fit WHERE pipeline = ?",
        (synthetic.blend(one, three).id,),
    )
    assert stands_on[0]["n"] == 3 * len(folds)
    assert _run(ledger, evaluation, synthetic.blend(one, three)).run == ""
    assert _run(
        ledger, evaluation, synthetic.blend(one, three, shrink=0.5)
    ).fits_computed == len(folds)


def test_pickle_is_marked_not_portable_and_an_unknown_format_is_refused(
    ledger, dataset, evaluation
):
    pickled = dataclasses.replace(
        synthetic.ridge(1), save=synthetic.pickle_save, load=synthetic.pickle_load
    )
    _run(ledger, evaluation, pickled)
    assert ledger.sql("SELECT portable FROM fit")[0]["portable"] == 0
    with pytest.raises(Refused, match="'zip', not one of"):
        _run(ledger, evaluation, dataclasses.replace(pickled, save=synthetic.zip_save))


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
        "from ml_lab.experiment import scorer\n"
        "from steps_w import TARGET, sim_metrics\n\n"
        '@scorer(metrics=sim_metrics, directions={"pnl": "max", "turnover": "min"})\n'
        "def sim(pred, session, rng, config):\n"
        "    truth = session.column(TARGET, rng)\n"
        "    flips = np.abs(np.diff(np.sign(pred), prepend=0.0))\n"
        "    return np.column_stack([np.sign(pred) * truth - 0.001 * flips, flips])\n"
    )
    module = cli._load(str(steps))
    evaluation = dataclasses.replace(
        module.evaluation(module.dataset(ledger)), scorer=cli._load(str(scoring)).sim
    )
    runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    first = _latest(ledger, evaluation)["ridge_3m"]
    scoring.write_text(scoring.read_text().replace("0.001 * flips", "0.002 * flips"))
    changed = dataclasses.replace(evaluation, scorer=cli._load(str(scoring)).sim)
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
    failure = ledger.sql("SELECT pipeline, error FROM failure")[0]
    assert failure["pipeline"] == flaky.id and "boom" in failure["error"]
    event = ledger.events(Event.FAILED, key=flaky.id)[-1]
    assert (
        "RuntimeError" in ledger.get_blob(event["payload"]["traceback"]["sha"]).decode()
    )
    again = _run(ledger, evaluation, flaky)
    assert again.fits_computed == len(folds) - 2 and again.scores_recorded == 1
    assert set(_latest(ledger, evaluation)) == {"flaky", "ridge_1m"}


def test_a_fit_that_imports_a_distribution_lazily_is_refused():
    pipeline = synthetic.ridge(3)
    shas = identity.import_shas(pipeline.steps, REPO)
    dists = identity.imported_dists(pipeline.steps, REPO)
    before_pytest = {name for name in sys.modules if not name.startswith("pytest")}
    with pytest.raises(Refused, match="distributions \\['pytest'\\]"):
        runs._refuse_lazy_imports(pipeline, REPO, shas, dists, before_pytest)


def test_the_run_records_the_resolution_file_when_present(ledger, evaluation):
    _run(ledger, evaluation, synthetic.ridge(3))
    run = ledger.events(Event.RUN)[0]["payload"]
    assert run["resolution"] is None or run["resolution"]["path"].startswith(
        ("uv.lock", "requirements")
    )


# Command line =========================================================================


def test_fy_run_merges_pipelines_from_several_modules_and_defaults_the_dataset(
    ledger, dataset, evaluation, tmp_path, capsys
):
    extra = tmp_path / "agent7.py"
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
    assert (
        cli.main(["--root", str(ledger.root), "run", by_path, "--dataset", dataset[:6]])
        == 0
    )
    assert "up to date" in capsys.readouterr().out


def test_fy_run_refuses_bad_experiments(ledger, dataset, tmp_path, capsys):
    extra = tmp_path / "bare.py"
    extra.write_text("from tests.synthetic import ridge\npipelines = [ridge(12)]\n")
    root = ["--root", str(ledger.root)]
    assert cli.main([*root, "run", str(extra)]) == 1
    assert "0 evaluations" in capsys.readouterr().err
    assert cli.main([*root, "run", "nope.decl"]) == 1
    assert "No module named 'nope'" in capsys.readouterr().err
    assert cli.main([*root, "run", "tests.synthetic", "--dataset", "zzz"]) == 1
    assert "0 matches" in capsys.readouterr().err


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


def test_fy_run_refuses_without_a_dataset(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path / "empty"), "run", "tests.synthetic"]) == 1
    assert "lab ingest first" in capsys.readouterr().err


# Read back ============================================================================


def test_the_views_read_with_sqlite_alone(ledger, dataset, evaluation, tmp_path):
    folds = evaluation.split.folds(load(ledger, dataset))
    _run(ledger, evaluation, synthetic.ridge(1), synthetic.ridge(6))
    db = sqlite3.connect(tmp_path / "ml-lab" / "ml_lab.sqlite")
    metrics = len(evaluation.directions)
    assert db.execute("SELECT COUNT(*) FROM fold_score").fetchone()[0] == (
        2 * len(folds) * len(evaluation.split.horizons) * metrics
    )
    assert db.execute("SELECT COUNT(*) FROM aggregate_score").fetchone()[0] == (
        2 * len(evaluation.split.horizons) * metrics
    )
    assert db.execute("SELECT COUNT(*) FROM latest_score").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM fit").fetchone()[0] == 2 * len(folds)
    assert db.execute("SELECT COUNT(*) FROM score_fit").fetchone()[0] == 2 * len(folds)
    assert (
        db.execute("SELECT SUM(duration_s) FROM score_fit").fetchone()
        == db.execute("SELECT SUM(duration_s) FROM fit").fetchone()
    )
    assert db.execute("SELECT DISTINCT source FROM latest_score").fetchall() == [
        ("synthetic",)
    ]
    scored, folds_, std, sha = db.execute(
        "SELECT score, folds, fold_std, series FROM aggregate_score WHERE window = '1'"
    ).fetchone()
    assert folds_ == len(folds) and std > 0
    stored = ledger.latest(Event.SCORE, scored)["payload"]["series"]["1"]
    assert stored["sha"] == sha
    assert len(_load_series(ledger, sha)) == sum(stored["fold_rows"])
    assert db.execute("SELECT resolution, pipelines FROM run").fetchone()[1]


def test_the_log_reads_as_it_stood(ledger, evaluation):
    _run(ledger, evaluation, synthetic.ridge(6))
    before = ledger.events()[-1]["seq"]
    _run(ledger, evaluation, synthetic.ridge(1))
    rows = ledger.sql("SELECT COUNT(*) AS n FROM score WHERE seq <= ?", (before,))
    assert rows[0]["n"] == 1 and len(ledger.events(Event.SCORE)) == 2


# Storage ==============================================================================


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


# Panel ================================================================================


def test_a_panel_grids_a_session_by_time_and_key_and_back():
    ts = np.array(["2025-01-01"] * 2 + ["2025-01-02"], dtype="datetime64[s]")
    session = Session(
        {"stock": np.array([1, 2, 1]), "px": np.array([10.0, 20.0, 11.0])}, ts
    )
    panel = Panel(session, "stock")
    grid = panel.grid("px")
    assert panel.shape == (2, 2) and np.isnan(grid[1, 1])
    assert np.array_equal(grid[:, 0], [10.0, 11.0])
    assert np.array_equal(panel.rows(grid), session.columns["px"])
    demeaned = grid - np.nanmean(grid, axis=1, keepdims=True)
    assert np.allclose(panel.rows(demeaned), [-5.0, 5.0, 0.0])
    assert Panel(session.upto(2), "stock").shape == (1, 2)
    with pytest.raises(ValueError, match="timestamps"):
        Panel(Session({"stock": np.array([1])}), "stock")
