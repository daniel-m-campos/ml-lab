"""The ledger story on synthetic data: ingest, declare, run, decide, read back.

Mirrors docs/user-stories.md; every event type in docs/spec.md is written and read back.
"""

from __future__ import annotations

import dataclasses
import io
import pathlib
import sqlite3
import sys

import pyarrow.parquet as pq
import pytest

from forestry import cli, data, decisions, hashing, review, runs
from forestry.ledger import Event, Ledger, Refused
from tests import synthetic

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "forestry")


@pytest.fixture
def dataset(ledger: Ledger) -> str:
    return synthetic.dataset(ledger)


@pytest.fixture
def evaluation(dataset: str):
    return synthetic.evaluation(dataset)


def _run(ledger, evaluation, *pipelines):
    return runs.run(ledger, list(pipelines), evaluation, code_root=REPO)


def _seat(ledger, evaluation, pipeline) -> str:
    _run(ledger, evaluation, pipeline)
    decisions.decide(ledger, pipeline.id, evaluation, kind="promote", why="incumbent")
    return pipeline.id


def _types(ledger) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in ledger.events():
        out[e["type"]] = out.get(e["type"], 0) + 1
    return out


# Dataset ==========================================================================================


def test_ingesting_the_same_recipe_twice_appends_one_event(ledger):
    assert synthetic.dataset(ledger) == synthetic.dataset(ledger)
    assert _types(ledger) == {Event.DATASET: 1}


def test_the_dataset_blob_opens_with_pyarrow_alone(ledger, dataset):
    event = ledger.latest(Event.DATASET, dataset)
    assert event["payload"]["blob"]["format"] == "parquet"
    table = pq.read_table(io.BytesIO(ledger.get_blob(event["payload"]["blob"]["sha"])))
    assert synthetic.TARGET in table.column_names and table.num_rows == event["payload"]["rows"]
    assert ledger.sql("SELECT process, instrument FROM dataset")[0] == {
        "process": "synthetic",
        "instrument": "SYN",
    }


# Declarations =====================================================================================


def test_a_pipeline_name_is_a_label_and_a_config_change_is_a_new_id():
    a = synthetic.ridge(3)
    assert a.named("other").id == a.id
    assert a.with_config(alpha=2.0).id != a.id


def test_any_evaluation_field_change_is_a_new_evaluation_id(dataset, evaluation):
    cost = synthetic.evaluation(dataset, cost=0.01)
    ages = dataclasses.replace(evaluation, ages=(1,))
    assert len({evaluation.id, cost.id, ages.id}) == 3


def test_the_schedule_is_embargoed_and_stored_once_in_the_evaluation_event(
    ledger, dataset, evaluation
):
    session = data.session(ledger, dataset)
    folds = runs.expand(evaluation, session)
    assert len(folds) >= evaluation.min_folds
    for fold in folds:
        gap = session.ts[fold.evals[1][0]] - session.ts[fold.train[1] - 1]
        assert gap.astype(int) >= evaluation.embargo_seconds
    _run(ledger, evaluation, synthetic.ridge(3))
    stored = ledger.latest(Event.EVALUATION, evaluation.id)["payload"]
    assert len(stored["folds"]) == len(folds) and stored["metrics"] == evaluation.directions


# Run ==============================================================================================


def test_a_run_posts_fits_predictions_and_one_entry_and_a_rerun_writes_nothing(
    ledger, dataset, evaluation
):
    folds = runs.expand(evaluation, data.session(ledger, dataset))
    first = _run(ledger, evaluation, synthetic.ridge(3))
    assert first.fits_computed == len(folds)
    assert first.predictions_computed == len(folds) * len(evaluation.ages)
    assert first.entries_scored == 1
    before = len(ledger.events())
    second = _run(ledger, evaluation, synthetic.ridge(3))
    assert (second.fits_computed, second.predictions_computed, second.entries_scored) == (0, 0, 0)
    assert (second.fits_reused, second.predictions_reused, second.entries_existing) == (
        len(folds),
        len(folds) * len(evaluation.ages),
        1,
    )
    assert second.run == "" and ledger.events()[before:] == []


def test_adding_a_pipeline_costs_only_its_own_fits(ledger, dataset, evaluation):
    folds = runs.expand(evaluation, data.session(ledger, dataset))
    _run(ledger, evaluation, synthetic.ridge(3))
    report = _run(ledger, evaluation, synthetic.ridge(3), synthetic.ridge(6))
    assert report.fits_computed == len(folds) and report.entries_scored == 1
    assert report.fits_reused == len(folds) and report.entries_existing == 1
    assert len(ledger.events(Event.RUN)) == 2


def test_fits_land_on_the_dataset_stream_and_entries_on_the_evaluation_stream(
    ledger, dataset, evaluation
):
    _run(ledger, evaluation, synthetic.ridge(3))
    assert {e["stream"] for e in ledger.events(Event.FIT)} == {dataset}
    assert {e["stream"] for e in ledger.events(Event.PREDICTIONS)} == {dataset}
    assert {e["stream"] for e in ledger.events(Event.ENTRY)} == {evaluation.id}


def test_a_scoring_change_refits_nothing_and_writes_a_new_entry(ledger, dataset, evaluation):
    _run(ledger, evaluation, synthetic.ridge(3))
    report = _run(ledger, synthetic.evaluation(dataset, cost=0.01), synthetic.ridge(3))
    assert report.fits_computed == 0 and report.entries_scored == 1
    assert len(ledger.events(Event.ENTRY)) == 2


def test_a_fit_carries_code_identity_and_its_model_reloads_without_pickle(ledger, evaluation):
    pipeline = synthetic.ridge(3)
    _run(ledger, evaluation, pipeline)
    fit = ledger.events(Event.FIT)[0]
    payload = fit["payload"]
    assert "tests/synthetic.py" in payload["import_shas"]
    assert payload["model"]["format"] == "arrow-arrays"
    lock = dict(
        line.split("==", 1) for line in ledger.get_blob(payload["env_lock"]["sha"]).decode().split()
    )
    assert {"python", "numpy", "pyarrow"} <= set(lock) and "pytest" not in lock
    assert lock["forestry"].count("+") == 1
    run = ledger.get(payload["run"])
    assert run["type"] == Event.RUN and "commit" in run["payload"]["git"]
    model = pipeline.load(ledger.get_blob(payload["model"]["sha"]))
    assert model.weights.shape == (3,)


def test_a_changed_source_file_is_a_new_fit_and_entry_but_the_same_pipeline(ledger, tmp_path):
    code = tmp_path / "steps_v.py"
    code.write_text((REPO / "tests" / "synthetic.py").read_text().replace("SYN", "SYNV"))
    module = cli._load(str(code))
    dataset = module.dataset(ledger)
    evaluation = module.evaluation(dataset)
    pipeline = module.ridge(3)
    runs.run(ledger, [pipeline], evaluation, code_root=tmp_path)
    decisions.decide(ledger, pipeline.id, evaluation, kind="promote", why="first")
    first_entry = decisions.baseline(ledger, evaluation)
    code.write_text(code.read_text().replace("0.01 * np.sum", "0.02 * np.sum"))
    module = cli._load(str(code))
    report = runs.run(ledger, [module.ridge(3)], evaluation, code_root=tmp_path)
    assert module.ridge(3).id == pipeline.id
    assert report.fits_computed > 0 and report.entries_scored == 1
    board = {r["pipeline"]: r for r in review.board(ledger, evaluation)}
    assert board["ridge_3m"]["status"] == "scored"
    assert decisions.baseline(ledger, evaluation) == first_entry


def test_a_failing_pipeline_is_recorded_and_the_rest_continue_and_a_rerun_resumes(
    ledger, dataset, evaluation
):
    folds = runs.expand(evaluation, data.session(ledger, dataset))
    synthetic.FLAKY_CALLS.clear()
    flaky = synthetic.flaky()
    report = _run(ledger, evaluation, flaky, synthetic.ridge(1))
    assert list(report.failed) == [flaky.id] and "boom" in report.failed[flaky.id]
    assert report.fits_computed == 2 + len(folds) and report.entries_scored == 1
    failure = ledger.events(Event.FAILED, key=flaky.id)[-1]
    assert "RuntimeError" in ledger.get_blob(failure["payload"]["traceback"]["sha"]).decode()
    board = {r["pipeline"]: r for r in review.board(ledger, evaluation)}
    assert board["flaky"]["status"] == "failed" and "boom" in board["flaky"]["why"]
    assert "traceback" in review.detail(ledger, evaluation, flaky.id)
    again = _run(ledger, evaluation, flaky)
    assert again.fits_computed == len(folds) - 2 and again.entries_scored == 1
    assert review.board(ledger, evaluation, {flaky.id})[0]["status"] == "scored"


def test_a_fit_that_imports_a_distribution_lazily_is_refused():
    pipeline = synthetic.ridge(3)
    shas = hashing.import_shas(pipeline.steps, REPO)
    dists = hashing.imported_dists(pipeline.steps, REPO)
    before_pytest = {name for name in sys.modules if not name.startswith("pytest")}
    with pytest.raises(Refused, match="distributions \\['pytest'\\]"):
        runs._refuse_lazy_imports(pipeline, REPO, shas, dists, before_pytest)


def test_the_board_shows_declared_pipelines_and_the_baseline(ledger, evaluation):
    _seat(ledger, evaluation, synthetic.ridge(1))
    _run(ledger, evaluation, synthetic.ridge(3), synthetic.ridge(6))
    names = [r["pipeline"] for r in review.board(ledger, evaluation, {synthetic.ridge(6).id})]
    assert names == ["ridge_1m", "ridge_6m"]
    assert len(review.board(ledger, evaluation)) == 3
    detail = review.detail(ledger, evaluation, synthetic.ridge(6).id)
    assert detail["env_diff"] == {} and detail["config_diff"]
    run = ledger.events(Event.RUN)[0]["payload"]
    assert run["resolution"] is None or run["resolution"]["path"].startswith(
        ("uv.lock", "requirements")
    )


# Decide ===========================================================================================


def test_decide_refuses_an_unscored_pipeline_and_promote_seats_a_baseline(ledger, evaluation):
    pipeline = synthetic.ridge(3)
    with pytest.raises(Refused):
        decisions.decide(ledger, pipeline.id, evaluation, kind="promote", why="x")
    _seat(ledger, evaluation, pipeline)
    entry = decisions.latest_entry(ledger, pipeline.id, evaluation)["id"]
    assert decisions.baseline(ledger, evaluation) == entry
    assert ledger.events(Event.DECISION)[0]["payload"]["against"] is None


def test_a_promotion_records_what_it_was_made_against(ledger, evaluation):
    _seat(ledger, evaluation, synthetic.ridge(6))
    challenger = synthetic.ridge(1)
    _run(ledger, evaluation, challenger)
    seen = decisions.compare(ledger, challenger.id, evaluation)
    assert set(seen["deltas"]) == set(evaluation.directions)
    decisions.decide(ledger, challenger.id, evaluation, kind="promote", why="pnl up")
    decision = ledger.events(Event.DECISION)[-1]
    assert decision["payload"]["against"] == seen and decision["actor"]
    assert (
        decisions.baseline(ledger, evaluation)
        == decisions.latest_entry(ledger, challenger.id, evaluation)["id"]
    )


def test_reject_records_why_and_leaves_the_baseline(ledger, evaluation):
    _seat(ledger, evaluation, synthetic.ridge(6))
    head = decisions.baseline(ledger, evaluation)
    loser = synthetic.ridge(1)
    _run(ledger, evaluation, loser)
    decisions.decide(ledger, loser.id, evaluation, kind="reject", why="too much turnover")
    assert decisions.baseline(ledger, evaluation) == head
    assert review.detail(ledger, evaluation, loser.id)["status"] == "rejected"


def test_a_pipeline_under_another_evaluation_is_not_decidable_here(ledger, dataset, evaluation):
    other = synthetic.evaluation(dataset, cost=0.01)
    _run(ledger, other, synthetic.ridge(1))
    with pytest.raises(Refused):
        decisions.decide(ledger, synthetic.ridge(1).id, evaluation, kind="promote", why="x")


def test_dominance_reads_directions():
    directions = {"pnl": "max", "turnover": "min"}
    better = decisions.dominance({"pnl": 2, "turnover": 1}, {"pnl": 1, "turnover": 2}, directions)
    worse = decisions.dominance({"pnl": 1, "turnover": 2}, {"pnl": 2, "turnover": 1}, directions)
    mixed = decisions.dominance({"pnl": 2, "turnover": 2}, {"pnl": 1, "turnover": 1}, directions)
    assert (better, worse, mixed) == ("dominates", "dominated", "incomparable")


# Review ===========================================================================================


def test_the_board_survives_a_rerun_and_answers_what_moved_the_baseline(ledger, evaluation):
    base = _seat(ledger, evaluation, synthetic.ridge(6))
    _run(ledger, evaluation, *synthetic.pipelines)
    _run(ledger, evaluation, *synthetic.pipelines)
    board = review.board(ledger, evaluation)
    assert board[0]["id"] == base and board[0]["status"] == "baseline"
    assert all(r["verdict"] and r["deltas"] for r in board[1:])
    assert {r["pipeline"] for r in board} == {"ridge_1m", "ridge_3m", "ridge_6m"}
    assert [h["against"] for h in review.history(ledger, evaluation)] == [None]
    detail = review.detail(ledger, evaluation, board[1]["id"])
    assert detail["config_diff"]["train_window_months"]["baseline"] == 6
    assert detail["folds"] and detail["status"] == "scored" and len(detail["entries"]) == 1
    decisions.decide(ledger, board[1]["id"], evaluation, kind="promote", why="better")
    assert review.detail(ledger, evaluation, base)["status"] == "superseded"


def test_the_views_read_with_sqlite_alone(ledger, evaluation, tmp_path):
    _seat(ledger, evaluation, synthetic.ridge(6))
    _run(ledger, evaluation, synthetic.ridge(1))
    db = sqlite3.connect(tmp_path / "forestry" / "forestry.sqlite")
    folds = db.execute("SELECT COUNT(*) FROM fold_score").fetchone()[0]
    assert (
        folds == 2 * len(ledger.get(decisions.baseline(ledger, evaluation))["payload"]["folds"]) * 2
    )
    assert db.execute("SELECT COUNT(*) FROM baseline").fetchone()[0] == 1
    statuses = dict(db.execute("SELECT pipeline, status FROM status").fetchall())
    assert sorted(statuses.values()) == ["baseline", "scored"]
    assert db.execute("SELECT COUNT(*) FROM board").fetchone()[0] == 2


def test_the_log_reads_as_it_stood(ledger, evaluation):
    first = _seat(ledger, evaluation, synthetic.ridge(6))
    before = ledger.events()[-1]["seq"]
    second = synthetic.ridge(1)
    _run(ledger, evaluation, second)
    decisions.decide(ledger, second.id, evaluation, kind="promote", why="later")
    assert (
        decisions.baseline(ledger, evaluation, upto=before)
        == decisions.latest_entry(ledger, first, evaluation)["id"]
    )


# Command line =====================================================================================


def test_fy_run_merges_pipelines_from_several_modules(ledger, dataset, tmp_path, capsys):
    extra = tmp_path / "agent7.py"
    extra.write_text("from tests.synthetic import ridge\npipelines = [ridge(12)]\n")
    by_path = str(REPO / "tests" / "synthetic.py")
    argv = ["--root", str(ledger.root), "run", by_path, str(extra), "--dataset", dataset]
    assert cli.main(argv) == 0
    assert "entries 4" in capsys.readouterr().out
    names = {r["pipeline"] for r in review.board(ledger, synthetic.evaluation(dataset))}
    assert names == {"ridge_1m", "ridge_3m", "ridge_6m", "ridge_12m"}


def test_fy_run_refuses_modules_without_exactly_one_evaluation(ledger, dataset, tmp_path, capsys):
    extra = tmp_path / "bare.py"
    extra.write_text("from tests.synthetic import ridge\npipelines = [ridge(12)]\n")
    assert cli.main(["--root", str(ledger.root), "run", str(extra), "--dataset", dataset]) == 1
    assert "0 evaluations" in capsys.readouterr().err
    assert cli.main(["--root", str(ledger.root), "run", "nope.decl", "--dataset", dataset]) == 1
    assert "No module named 'nope'" in capsys.readouterr().err


# Storage ==========================================================================================


def test_ulids_sort_and_do_not_collide():
    ids = [hashing.ulid() for _ in range(1000)]
    assert all(len(i) == 26 for i in ids) and len(set(ids)) == 1000 and ids == sorted(ids)


def test_blobs_are_written_atomically_under_their_sha(ledger):
    sha = ledger.put_blob(b"hello")
    assert (ledger.blobs / sha).exists() and ledger.get_blob(sha) == b"hello"
    assert sha == hashing.bytes_hash(b"hello")
    assert not list(ledger.blobs.glob(".tmp-*"))


def test_git_blob_sha_matches_git():
    assert hashing.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
