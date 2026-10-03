# forestry

A local-first record of experimentation on frozen, time-ordered datasets. Git owns the code, a content-addressed blob store owns the bytes, an append-only event log owns what happened and what was decided. Declare pipelines and an evaluation, run them memoized, read each one against the baseline, and move the baseline only through a recorded decision. The model is git plus code review: content-addressed objects, one movable baseline per evaluation, decisions as its reflog.

- Design: [docs/spec.md](docs/spec.md); concrete walkthroughs: [docs/user-stories.md](docs/user-stories.md)
- Landscape survey and build-versus-buy verdict: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md), notes under `research_notes/`

## Try it

```
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # the ledger story on synthetic data, tests/test_ledger_story.py
examples/optiver/campaign.sh 20         # real order-book data, ridge vs bonsai; needs data/optiver
```

The Optiver example (`examples/optiver/`) is the template. `campaign.sh` is a handful of `fy` commands (freeze, run, decide, board, history) over three per-project modules: `capture.py` (how to read the raw data), `steps.py` (features, models, scorers) and `declarations.py` (the pipelines and the evaluation). A new idea is one more `Pipeline` in `declarations.py` and `fy run` again. It needs the Kaggle "Trading at the Close" `train.csv` under `data/optiver/optiver-trading-at-the-close/`; `tests/test_optiver_example.py` skips without it.

Layout: `src/forestry/ledger.py` (the event table, its views, the blob store), `declare.py` (Pipeline, Evaluation, step, scorer), `harness.py` (folds, runs, fits memoized on import shas and the environment lock, scores, decide), `data.py` (freeze), `formats.py` (Parquet, Arrow IPC and temp-file round trips; no pickle), `hashing.py` (content hashes, ULIDs, git blob shas), `session.py` (the resident rows), `review.py` (board, detail, history), `cli.py` (`fy`).
