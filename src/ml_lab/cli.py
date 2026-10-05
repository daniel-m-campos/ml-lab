"""``lab``: the command line over the log, two verbs; reading is SQL over the views.

Declarations are plain module attributes, so a project may organize them freely: one
script, or ``dataset.py`` + ``steps.py`` + ``experiment.py``, or one file per idea. ``fy
ingest`` reads ``dataset(ledger, *args)`` from a module. ``lab run`` takes one or more
modules, reads ``pipelines`` (a list of ``Pipeline``) from each and ``evaluations`` (a
list of ``Evaluation``, or a function of the dataset id returning one) from exactly one
of them, so a file an agent wrote holding only new pipelines runs beside the project's
declarations; every pipeline is scored under every evaluation. A module
is a dotted name importable from the current directory or a ``.py`` path. The dataset
defaults to the newest one recorded. The ledger root comes from ``--root`` or
``ML_LAB_ROOT`` (default ``.ml-lab``); only ``lab ingest`` creates one, ``lab run``
refuses a root without a ledger. The actor comes from ``ML_LAB_ACTOR``.

Examples
--------
$ lab ingest examples/optiver/dataset.py 20
$ lab run examples/optiver/experiment.py
$ lab run examples/optiver/experiment.py ideas/agent7.py
$ sqlite3 -box .ml-lab/ml_lab.sqlite "SELECT * FROM score_latest"
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import pathlib
import sys
from typing import Any

from ml_lab import runs
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Ledger, Refused

DEFAULT_ROOT = ".ml-lab"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab")
    parser.add_argument("--root", default=os.environ.get("ML_LAB_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "ingest", help="call the module's dataset(ledger, *args); prints the id"
    )
    p.add_argument("module", help="module name or .py path exposing dataset()")
    p.add_argument("args", nargs="*")
    p.set_defaults(handler=_ingest)

    p = sub.add_parser("run", help="fit, predict and score every declared pipeline")
    p.add_argument(
        "experiments", nargs="+", help="modules or .py paths exposing pipelines"
    )
    p.add_argument(
        "--dataset",
        help="dataset id prefix or source name; default: the newest, when the ledger "
        "holds one source",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute nothing, write nothing; say what a run would compute and why",
    )
    p.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    try:
        if args.command == "run" and not (root / "ml_lab.sqlite").exists():
            raise Refused(
                f"no ledger at {root.resolve()}; set ML_LAB_ROOT or --root, or lab "
                "ingest to create one"
            )
        return args.handler(args, Ledger(root))
    except (Refused, KeyError, ImportError) as refused:
        print(f"lab {args.command}: {refused}", file=sys.stderr)
        return 1


# Handlers =============================================================================


def _ingest(args: argparse.Namespace, ledger: Ledger) -> int:
    print(_load(args.module).dataset(ledger, *args.args))
    return 0


def _run(args: argparse.Namespace, ledger: Ledger) -> int:
    """Every evaluation in turn, one shared plan; a pipeline that fails in one is
    skipped in the rest of this run and named once at the end, since failures are
    not memoized and a rerun retries it.
    """
    pipelines, evaluations = _experiments(args, ledger)
    failed: dict[str, str] = {}
    planned: set[str] = set()
    total = runs.RunReport()
    for evaluation in evaluations:
        print(f"evaluation {evaluation.id} {evaluation.name}".rstrip())
        live = [p for p in pipelines if (p.name or p.id) not in failed]
        if not live:
            print("skipped: every pipeline failed above")
            continue
        report = runs.run(
            ledger,
            live,
            evaluation,
            log=lambda line: print(line, flush=True),
            dry=args.dry_run,
            planned=planned,
        )
        for name, error in report.failed.items():
            print(f"lab run: {name} failed: {error}")
        failed |= report.failed
        for field in dataclasses.fields(runs.RunReport)[1:7]:
            vars(total)[field.name] += getattr(report, field.name)
        if args.dry_run:
            print(
                f"dry run: would compute fits {report.fits_computed}, predictions "
                f"{report.predictions_computed}, scores {report.scores_recorded}; "
                f"reuse {report.fits_reused}, {report.predictions_reused}, "
                f"{report.scores_reused}"
            )
        elif not report.run:
            print(
                f"up to date: {report.scores_reused} scores; {report.fits_reused} fits "
                f"and {report.predictions_reused} predictions reused, nothing written"
            )
        else:
            print(
                f"run {report.run}: fits {report.fits_computed}, predictions "
                f"{report.predictions_computed}, scores {report.scores_recorded}"
            )
    if len(evaluations) > 1:
        verb = "would compute" if args.dry_run else "computed"
        print(
            f"total over {len(evaluations)} evaluations: {verb} fits "
            f"{total.fits_computed}, predictions {total.predictions_computed}, scores "
            f"{total.scores_recorded}; reuse {total.fits_reused}, "
            f"{total.predictions_reused}, {total.scores_reused}"
        )
    if failed:
        print(f"failed, skipped in later evaluations: {', '.join(failed)}")
    return 1 if failed else 0


# Private Functions ====================================================================


def _experiments(
    args: argparse.Namespace, ledger: Ledger
) -> tuple[list[Pipeline], list[Evaluation]]:
    """Pipelines from every module named; the evaluations from the one module that
    declares them. Two names on one evaluation, or one name on two, are refused.
    """
    modules = [_load(spec) for spec in args.experiments]
    pipelines = [p for m in modules for p in getattr(m, "pipelines", [])]
    found = [m.evaluations for m in modules if hasattr(m, "evaluations")]
    if len(found) != 1:
        raise Refused(
            f"{len(found)} modules declare evaluations across {args.experiments}; "
            "need one"
        )
    evaluations = found[0]
    if callable(evaluations):
        evaluations = evaluations(_dataset(ledger, args.dataset))
    evaluations = list(evaluations)
    names, ids = {e.name or e.id for e in evaluations}, {e.id for e in evaluations}
    if not (evaluations and len(names) == len(ids) == len(evaluations)):
        raise Refused(
            f"{len(evaluations)} evaluations, {len(names)} names, {len(ids)} ids; one "
            "name per evaluation and one evaluation per name"
        )
    return pipelines, evaluations


def _dataset(ledger: Ledger, selector: str | None) -> str:
    """The dataset named by an id prefix or a source; with no selector, the newest,
    provided every dataset in the ledger shares one source. A source resolves to its
    newest dataset only while one actor wrote it: two studies under one source name
    are refused with their ids, since the newest would silently be someone else's.
    """
    rows = ledger.sql("SELECT id, source, actor FROM raw_dataset ORDER BY seq")
    by_source: dict[str, str] = {r["source"]: r["id"] for r in rows}
    if selector is None and len(by_source) == 1:
        selector = rows[-1]["source"]
    named = [r for r in rows if r["source"] == selector]
    if len({r["actor"] for r in named}) > 1:
        raise Refused(
            f"source {selector} names datasets by several actors, pass an id: "
            + ", ".join(f"{r['id'][:8]} ({r['actor']})" for r in named)
        )
    if selector in by_source:
        return by_source[selector]
    if selector:
        matches = [r["id"] for r in rows if r["id"].startswith(selector)]
        if len(matches) != 1:
            raise Refused(
                f"dataset {selector}: {len(matches)} matches in "
                f"{ledger.root.resolve()}; lab ingest first"
            )
        return matches[0]
    if len(by_source) > 1:
        choices = ", ".join(f"{s} ({i[:8]})" for s, i in by_source.items())
        raise Refused(f"ledger holds several sources, pass --dataset: {choices}")
    if not rows:
        raise Refused(
            f"dataset (newest): 0 matches in {ledger.root.resolve()}; lab ingest first"
        )
    return rows[-1]["id"]


def _load(spec: str) -> Any:
    """Import a dotted name from the current directory, or a .py path by its package
    name.
    """
    if spec.endswith(".py"):
        path = pathlib.Path(spec).resolve()
        parts = [path.stem]
        while (path.parent / "__init__.py").exists():
            path = path.parent
            parts.insert(0, path.name)
        sys.path.insert(0, str(path.parent))
        spec = ".".join(parts)
    else:
        sys.path.insert(0, os.getcwd())
    return importlib.import_module(spec)


if __name__ == "__main__":
    raise SystemExit(main())
