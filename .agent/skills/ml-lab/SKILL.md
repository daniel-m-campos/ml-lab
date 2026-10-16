---
name: ml-lab
description: How an agent runs an honest modelling study with ml-lab, the local experiment ledger for frozen datasets with a fixed row order. Use when asked to build, extend or judge a study that declares pipelines and evaluations for `lab run`, to add a candidate to an existing study, or to read a ledger's results. Covers the setup the identity check needs, the layout that keeps the memo cheap, the dry run habit, pre-registered picks, scoring test once, reading the board and head_to_head views by id, and recording frictions.
---

# Working a study with ml-lab

ml-lab records experimentation on a frozen dataset: `lab ingest` freezes rows, `lab run` fits, predicts and scores every declared pipeline, reuses what the ledger already holds, refuses the common leaks before writing, and stores every result as an event naming the code, data and environment behind it. Results are read with SQL. Read `docs/spec.md` end to end before the first run; it is the contract every rule below rests on.

## Setup the identity check needs

- Run `lab` from inside the git repository that holds the study packages; the tool hashes code under that repository root. Code reached from outside it must come from an installed distribution, editable or not, or the run is refused as code the memo cannot see. A study on `PYTHONPATH` alone fails that check, which is the first thing a newcomer hits: install it (`uv pip install -e .`) or keep it in the repository you run from.
- Set `ML_LAB_ROOT` to the ledger directory and `ML_LAB_ACTOR` to who is writing: your name, or `agent:<name>` for an agent. The actor is provenance, who recorded an event, not the study's name; a study is told apart by its dataset's `source`. When several actors ingested under one source, `--dataset <source>` resolves to your own newest dataset, and is refused with every actor's id when you recorded none; pass the id prefix then.
- Pass `--dataset <source name or id prefix>` on every run when the ledger holds several sources.
- An import inside a function body is refused, and so is a library that imports on first use: numba's first compile pulls in cffi, pycparser and charset_normalizer, and the fit or scorer that triggered it is refused as a lazy import. Warm such code up at module import, for example by calling the compiled function once on a tiny array at the bottom of its module.
- Install this skill once for every repository by linking it into your personal skills: `ln -s <ml-lab checkout>/.agent/skills/ml-lab ~/.claude/skills/ml-lab`, or copy the directory there. Studies live in their own repositories, which this file is not in.

## The shape of a study

- One directory per study, a Python package: `dataset.py` (the ingest), `experiment.py` (validation pipelines and the evaluations), `test.py` (the test evaluation, in its own module because a module's evaluations all run on every `lab run`), and `NOTES.md`. Work only inside the study you were given; sibling studies belong to other agents.
- Commands, run from the directory that holds the study packages:

```
lab ingest <study>/dataset.py [args]                      # prints the dataset id
lab run --dry-run <study>/experiment.py --dataset <src>   # what a run would compute and why
lab run <study>/experiment.py --dataset <src>             # validation; prints each pipeline id, then the run id
lab run <study>/test.py --dataset <src>                   # test, once, after the pick is written
```

## Layout that keeps the memo cheap

A fit is reused when the ledger holds one for the same data, declaration, code and environment. Its code key covers the function's module and every module it imports, so:

- Put every new pipeline, step and params class in a new module (`<study>/round<N>.py`, `<study>/round<N>_features.py`) and import the old ones. Editing a shared module refits every pipeline that imports it.
- Never change a parameter of an existing pipeline to try an idea; parameters are part of the pipeline id. Declare a new pipeline, with `dataclasses.replace` when it differs in one field.
- Keep constants and pipeline families in their own modules; a comment edit refits nothing, a code edit refits what imports it.
- Feature column order is part of a fit's identity, and it changes results in a booster: in Bonsai the order decides which features `feature_fraction` samples and where monotone constraints land. Treat the order as a declared choice.
- Run `lab run --dry-run` before every run and read its per-fit reasons. A refit of a house pipeline you did not expect is a layout mistake, not a data change; stop and find it before spending the time. A dry run also computes a new feature set, probes it on its first span so a leak it reads is refused there, and keeps it under the ledger's `planned` directory for the run that follows.

## Writing steps

- Everything learned lives inside `fit`. A category frequency, a scaler, an encoding or a round count chosen on data belongs in `fit` or in a probed `features` step, never precomputed over the whole dataset outside the tool. Early stopping is the same rule: cut the fold's train segments inside `fit` into an inner train and validation, choose the round count there, and refit on the whole train range; the tool has no helper for it yet (issue #23).
- Features are computed by a `features` function (or a tuple of them) and read through `dataset.column` and `dataset.matrix` by the names in `dataset.computed_columns`. The tool reruns every step on a prefix of the rows and refuses one whose output changes, so read the spec's paragraphs on scopes, probes and reveal lags before writing one.
- Declare each label's reveal lag and derive the embargo from it; the tool refuses a fold that trains on labels revealed after its test range starts, and names the purge that clears it.
- Normalise per timestamp, never over the whole predicted range; a cross-sectional statistic reads only the universe known at its row.
- A scorer returns one row per scoring unit, and every metric is a finite number even on a constant or empty series.

## Judging honestly

- Fit the rule baselines first (zero, majority class, a one-feature rule, a linear model); one of them may be the thing to beat.
- Before the first model, write the pick rule into `NOTES.md`: the reference, the primary metric and evaluation, the eligible candidates by name, the threshold (a paired t against the reference on `head_to_head`, with the direction), and how a more complex candidate replaces a simpler qualifier. Count the candidates against the threshold and state the family-wise false-pick chance. Anything added later is exploratory and cannot be picked.
- Pair every comparison by fold or by day. Fold spread measures regime spread between folds, not the noise of a difference; the paired delta's standard error does.
- Score the test evaluation once, at the end, on the reference and the qualifiers only, after the pick is written. If nothing qualifies, the pick is the reference and the test run is skipped. A holdout used to choose is no longer a holdout.
- Prefer two well-separated ideas with a mechanism to a sweep: a feature with a domain reason, a target or post-process the metric rewards, a blend of members that actually differ, a seed set when one seed's spread exceeds the effect. Test the ideas built on fits already in the ledger first; a blend or post-process over memoized fits costs nothing.
- Stay blind to leaderboards, public notebooks and published solutions unless the study is explicitly a reproduction.

## Reading results

Read the ledger with SQL, through its views, never by parsing logs. A name is a label, not an id: an edited pipeline gets a new id under the same name and the views list both, so filter on the pipeline ids `lab run` printed (also in `event_run.pipelines` under the run id it printed), or on `current = 1`, the ids the newest run that listed each name declared, and on the dataset's `source` to keep to one study.

```
sqlite3 -readonly $ML_LAB_ROOT/ml_lab.sqlite "SELECT evaluation_name, name, metric, fold_mean, fold_std, n_folds, pooled, rank, pipeline FROM board WHERE source = '<source>' AND current = 1"
sqlite3 -readonly $ML_LAB_ROOT/ml_lab.sqlite "SELECT evaluation_name, reference_name, name, metric, delta_mean, delta_std, wins, n_folds, t, pooled_delta FROM head_to_head WHERE source = '<source>' AND reference = '<reference id>' AND current = 1"
```

`board` holds one row per pipeline, evaluation and metric; `head_to_head` holds the paired fold differences between a pipeline and a reference, with the paired t. `event_fit`, `event_feature`, `event_prediction`, `event_score` and `event_run` are the building blocks, and a run records the commit, dirty diff, host and the tool's commit.

A number reproduces on the host that produced it; across hosts, boosters and solvers differ beyond the fourth digit, and a dataset id can differ across operating systems when the ingest does float math. Compare within one ledger on one host.

## Notes and frictions

- `NOTES.md` opens with a "How to run" block (the exact ingest and run commands with `--dataset`), then the pre-registered rule, then a results table (validation and test, reference and candidates, folds, paired t), then what was learned, then the frictions recorded. A later round adds a dated section and leaves earlier ones as they were.
- A friction is anything that cost more than a few minutes or left you unsure what the tool did: a refusal you did not expect, a missing view column, a step you had to write that the tool should own, docs that misled you. Record each one in `NOTES.md` with what you expected. File it as an issue in the tool's repository only when the repository is public and the repro is synthetic: never a private study's names, paths, columns or numbers. When in doubt, leave the filing to a human.
