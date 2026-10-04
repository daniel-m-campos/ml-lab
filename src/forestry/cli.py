"""``fy``: the command line over the log, two verbs; reading is SQL over the views.

Declarations are plain module attributes, so a project may organize them freely: one script, or
``capture.py`` + ``steps.py`` + ``declarations.py``, or one file per idea. ``fy ingest`` reads
``dataset(ledger, *args)`` from a module. ``fy run`` takes one or more modules, reads ``pipelines``
(a list of ``Pipeline``) from each and ``evaluation`` (an ``Evaluation`` or a function of the
dataset id) from exactly one of them, so an agent-written file holding only new pipelines runs
beside the project's declarations. A module is a dotted name importable from the current directory
or a ``.py`` path. The dataset defaults to the newest one recorded. The ledger root comes from
``--root`` or ``FORESTRY_ROOT`` (default ``.forestry``); the actor from ``FORESTRY_ACTOR``.

Examples
--------
$ fy ingest examples/optiver/capture.py 20
$ fy run examples/optiver/declarations.py
$ fy run examples/optiver/declarations.py ideas/agent7.py
$ sqlite3 -box .forestry/forestry.sqlite "SELECT * FROM latest_entry"
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
from typing import Any

from forestry import runs
from forestry.declare import Evaluation, Pipeline
from forestry.ledger import Event, Ledger, Refused

DEFAULT_ROOT = ".forestry"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=os.environ.get("FORESTRY_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="call the module's dataset(ledger, *args); prints the id")
    p.add_argument("module", help="module name or .py path exposing dataset()")
    p.add_argument("args", nargs="*")
    p.set_defaults(handler=_ingest)

    p = sub.add_parser("run", help="fit, predict and score every declared pipeline")
    p.add_argument("declarations", nargs="+", help="modules or .py paths exposing pipelines")
    p.add_argument("--dataset", help="dataset id or prefix; default: the newest recorded")
    p.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    ledger = Ledger(pathlib.Path(args.root))
    try:
        return args.handler(args, ledger)
    except (Refused, KeyError, ImportError) as refused:
        print(f"fy {args.command}: {refused}", file=sys.stderr)
        return 1


# Handlers =========================================================================================


def _ingest(args: argparse.Namespace, ledger: Ledger) -> int:
    print(_load(args.module).dataset(ledger, *args.args))
    return 0


def _run(args: argparse.Namespace, ledger: Ledger) -> int:
    pipelines, evaluation = _declarations(args, ledger)
    report = runs.run(ledger, pipelines, evaluation)
    if not report.run:
        print(
            f"up to date: {report.entries_existing} entries; {report.fits_reused} fits and "
            f"{report.predictions_reused} predictions reused, nothing written"
        )
        return 0
    print(
        f"run {report.run}: fits {report.fits_computed}, predictions "
        f"{report.predictions_computed}, entries {report.entries_scored}"
    )
    names = {p.id: p.name or p.id for p in pipelines}
    for pipeline_id, error in report.failed.items():
        print(f"fy run: {names[pipeline_id]} failed: {error}", file=sys.stderr)
    return 1 if report.failed else 0


# Private Functions ================================================================================


def _declarations(args: argparse.Namespace, ledger: Ledger) -> tuple[list[Pipeline], Evaluation]:
    """Pipelines from every module named; the evaluation from the one module that declares it."""
    modules = [_load(spec) for spec in args.declarations]
    pipelines = [p for m in modules for p in getattr(m, "pipelines", [])]
    found = {id(m.evaluation): m.evaluation for m in modules if hasattr(m, "evaluation")}
    if len(found) != 1:
        raise Refused(f"{len(found)} evaluations declared across {args.declarations}; need one")
    evaluation = next(iter(found.values()))
    if callable(evaluation):
        evaluation = evaluation(_dataset(ledger, args.dataset))
    return pipelines, evaluation


def _dataset(ledger: Ledger, prefix: str | None) -> str:
    """The newest recorded dataset, or the one dataset whose id starts with ``prefix``."""
    ids = [e["id"] for e in ledger.events(Event.DATASET)]
    matches = [i for i in ids if i.startswith(prefix)] if prefix else ids[-1:]
    if len(matches) != 1:
        raise Refused(f"dataset {prefix or '(newest)'}: {len(matches)} matches; fy ingest first")
    return matches[-1]


def _load(spec: str) -> Any:
    """Import a dotted name from the current directory, or a .py path by its package name."""
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
