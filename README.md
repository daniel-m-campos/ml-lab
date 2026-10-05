# ml-lab

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. Declare pipelines and evaluations, run them memoized on code, data and environment identity, and read the results with SQL.

- The log: [docs/spec.md](docs/spec.md), the event and view tables, blob formats, declined designs and what is deferred
- Landscape survey and build-versus-buy verdict: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md), notes under `research_notes/`

## Install

```
git clone git@github.com:daniel-m-campos/ml-lab.git
uv pip install -e ml-lab                 # editable, into the project's venv; Python 3.12, numpy and polars
```

Editable is the point: the environment lock hashes the tool's code, so a `git pull` that changes how a fit is computed refits what it touched and a pull that changes docs refits nothing. A non-editable install carries only the version string and would reuse fits across tool changes. Upgrading is `git pull`; a ledger survives it unless the schema version moves, which `lab` refuses loudly with the instruction to delete and rerun. Every run records the tool's commit and dirty diff (`SELECT editable FROM raw_run`), so an issue can name the exact tool it hit. Feedback goes in as a GitHub issue with the `friction` label, with a minimal repro; fixes land by pull request and bump the patch version.

## Try it

```
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # the ledger story on synthetic data, tests/test_ledger_story.py
examples/optiver/launch.sh 20           # real order-book data, ridge vs bonsai; needs data/optiver
```

The Optiver example (`examples/optiver/`) is the template. `launch.sh` is `lab ingest`, `lab run` and one SQL query over three per-project modules: `dataset.py` (how to read the raw data), `steps.py` (features, models, scorers) and `experiment.py` (the pipelines and the evaluations). Modules are given to `lab` as paths or as dotted names from `examples/`. A new idea is one more `Pipeline` in `experiment.py`, or a new file holding only pipelines, and `lab run` again. It needs the Kaggle "Trading at the Close" `train.csv` under `data/optiver/optiver-trading-at-the-close/`; `tests/test_optiver_example.py` skips without it.

Layout: `src/ml_lab/ledger.py` (the event table, its views, the blob store), `experiment.py` (Pipeline, Evaluation, step, scorer), `splits.py` (WalkForward, BlockedKFold, Holdout by rows; CalendarWalkForward on the clock), `dates.py` (month arithmetic), `runs.py` (fits memoized on code keys plus the environment lock, import shas recorded; scores, failures), `dataset.py` (record, load), `formats.py` (Parquet, Arrow IPC and temp-file round trips; pickle only as non-portable), `identity.py` (content hashes, ULIDs, git blob shas, import closures), `session.py` (the resident rows), `panel.py` (a (time, key) grid over the rows), `cli.py` (`lab`).
