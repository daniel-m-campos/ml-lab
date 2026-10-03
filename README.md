# forestry

A local-first ledger for the lifecycle of experimentation on frozen datasets: compose pipelines, compare candidates under a declared evaluation, move a baseline only through a recorded comparison, deploy, refresh. Agents and humans write to the same ledger.

- Design: [docs/spec.md](docs/spec.md); concrete walkthroughs: [docs/user-stories.md](docs/user-stories.md)
- Landscape survey and build-versus-buy verdict: [reports/Forestry MLOps landscape survey.md](reports/Forestry%20MLOps%20landscape%20survey.md), notes under `research_notes/`

## Try it

```
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest          # the campaign story, tests/test_campaign_story.py
.venv/bin/python examples/toy_campaign.py   # one campaign on synthetic data, printing the board
.venv/bin/python examples/optiver/campaign.py --stocks 20   # real order-book data, ridge vs bonsai; needs data/optiver
```

The Optiver example (`examples/optiver/`) needs the Kaggle "Trading at the Close" `train.csv` under `data/optiver/optiver-trading-at-the-close/`; `tests/test_optiver_example.py` skips without it.

Layout: `src/forestry/declare.py` (Pipeline, Evaluation, Stage, gates), `harness.py` (schedule, memoized fits and predictions, stages, gates, seal, deploy), `ledger.py` (SQLite rows plus blobs), `data.py` (freeze), `review.py` (board, history, why, find), `toy.py` (synthetic data, toy pipelines and scorers), `cli.py` (`fy`).
