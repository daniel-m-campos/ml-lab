# forestry

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened. Declare pipelines and an evaluation, run them memoized on code, data and environment identity, and read the results with SQL.

- Design: [docs/spec.md](docs/spec.md); concrete walkthroughs: [docs/user-stories.md](docs/user-stories.md)
- Landscape survey and build-versus-buy verdict: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md), notes under `research_notes/`

## Try it

```
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # the ledger story on synthetic data, tests/test_ledger_story.py
examples/optiver/launch.sh 20           # real order-book data, ridge vs bonsai; needs data/optiver
```

The Optiver example (`examples/optiver/`) is the template. `launch.sh` is `fy ingest`, `fy run` and one SQL query over three per-project modules: `dataset.py` (how to read the raw data), `steps.py` (features, models, scorers) and `experiment.py` (the pipelines and the evaluation). Modules are given to `fy` as paths or as dotted names from `examples/`. A new idea is one more `Pipeline` in `experiment.py`, or a new file holding only pipelines, and `fy run` again. It needs the Kaggle "Trading at the Close" `train.csv` under `data/optiver/optiver-trading-at-the-close/`; `tests/test_optiver_example.py` skips without it.

Layout: `src/forestry/ledger.py` (the event table, its views, the blob store), `experiment.py` (Pipeline, Evaluation, step, scorer), `splits.py` (WalkForward, BlockedKFold, Holdout), `runs.py` (fits memoized on import shas and the environment lock, scores, failures), `dataset.py` (record, load), `formats.py` (Parquet, Arrow IPC and temp-file round trips; no pickle), `identity.py` (content hashes, ULIDs, git blob shas, import closures), `session.py` (the resident rows), `cli.py` (`fy`).
