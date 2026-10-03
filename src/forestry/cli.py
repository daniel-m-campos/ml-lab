"""``fy``: the command line over the log.

Declarations are plain module attributes, so a project may organize them freely: one script, or
``capture.py`` + ``steps.py`` + ``declarations.py``, or one file per idea. ``fy ingest`` reads
``dataset(ledger, *args)`` from a module. The other verbs take one or more modules and read
``pipelines`` (a list of ``Pipeline``) from each and ``evaluation`` (an ``Evaluation`` or a
function of the dataset id) from exactly one of them, so an agent-written file holding only new
pipelines runs beside the project's declarations. Every one takes ``--dataset``. The ledger root
comes from ``--root`` or ``FORESTRY_ROOT`` (default ``.forestry``); the actor from
``FORESTRY_ACTOR``. Pipelines are named by name or id prefix.

Examples
--------
$ DS=$(fy ingest optiver.capture 20)
$ C="optiver.declarations --dataset $DS"
$ fy run $C
$ fy decide $C ridge_3m --kind promote --why "the model in production"
$ fy board $C
$ fy run optiver.declarations ideas/agent7.py --dataset $DS
$ fy history $C
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import pathlib
import sys
from typing import Any

from forestry import decisions, review, runs
from forestry.declare import Evaluation, Pipeline
from forestry.ledger import Event, Ledger, Refused

DEFAULT_ROOT = ".forestry"
BOARD_COLUMNS = ("entry", "pipeline", "status", "verdict", "deltas", "metrics", "why")
HISTORY_COLUMNS = ("id", "pipeline", "key", "against", "why", "actor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=os.environ.get("FORESTRY_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="call the module's dataset(ledger, *args); prints the id")
    p.add_argument("module", help="module name or .py path exposing dataset()")
    p.add_argument("args", nargs="*")
    p.set_defaults(handler=_ingest)

    p = sub.add_parser("run", help="fit, predict and score every pipeline")
    _declarations_arguments(p)
    p.set_defaults(handler=_run)

    p = sub.add_parser("board", help="every pipeline against the baseline; one name for detail")
    _declarations_arguments(p)
    p.add_argument("pipeline", nargs="?", help="one pipeline's detail as JSON")
    p.set_defaults(handler=_board)

    p = sub.add_parser("decide", help="record promote or reject on a pipeline")
    _declarations_arguments(p)
    p.add_argument("pipeline")
    p.add_argument("--kind", required=True, choices=["promote", "reject"])
    p.add_argument("--why", required=True)
    p.set_defaults(handler=_decide)

    p = sub.add_parser("history", help="the chain of promotions")
    _declarations_arguments(p)
    p.set_defaults(handler=_history)

    args = parser.parse_args(argv)
    ledger = Ledger.open(pathlib.Path(args.root))
    try:
        return args.handler(args, ledger)
    except (Refused, KeyError) as refused:
        print(f"fy {args.command}: {refused}", file=sys.stderr)
        return 1


# Handlers =========================================================================================


def _declarations_arguments(p: argparse.ArgumentParser):
    p.add_argument("declarations", nargs="+", help="modules or .py paths exposing pipelines")
    p.add_argument("--dataset", required=True, help="dataset id from fy ingest")


def _ingest(args: argparse.Namespace, ledger: Ledger) -> int:
    print(_load(args.module).dataset(ledger, *args.args))
    return 0


def _run(args: argparse.Namespace, ledger: Ledger) -> int:
    pipelines, evaluation = _declarations(args)
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
    return 0


def _board(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _declarations(args)
    if args.pipeline:
        pipeline = _resolve(ledger, evaluation, args.pipeline)
        print(json.dumps(review.detail(ledger, evaluation, pipeline), indent=2, default=str))
    else:
        print(_table(BOARD_COLUMNS, review.board(ledger, evaluation)))
    return 0


def _decide(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _declarations(args)
    pipeline = _resolve(ledger, evaluation, args.pipeline)
    print(decisions.decide(ledger, pipeline, evaluation, kind=args.kind, why=args.why))
    return 0


def _history(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _declarations(args)
    print(_table(HISTORY_COLUMNS, review.history(ledger, evaluation)))
    return 0


# Private Functions ================================================================================


def _resolve(ledger: Ledger, evaluation: Evaluation, name: str) -> str:
    """A pipeline id from its name or an id prefix, among those with an entry here."""
    entries = ledger.events(Event.ENTRY, stream=evaluation.id)
    scored = sorted({e["payload"]["pipeline"] for e in entries})
    matches = [p for p in scored if p.startswith(name)]
    if not matches:
        matches = [p for p in scored if ledger.get(p)["payload"]["name"] == name]
    if len(matches) != 1:
        raise Refused(f"pipeline {name!r}: {len(matches)} matches under this evaluation")
    return matches[0]


def _declarations(args: argparse.Namespace) -> tuple[list[Pipeline], Evaluation]:
    """Pipelines from every module named; the evaluation from the one module that declares it."""
    modules = [_load(spec) for spec in args.declarations]
    pipelines = [p for m in modules for p in getattr(m, "pipelines", [])]
    found = {id(m.evaluation): m.evaluation for m in modules if hasattr(m, "evaluation")}
    if len(found) != 1:
        raise Refused(f"{len(found)} evaluations declared across {args.declarations}; need one")
    evaluation = next(iter(found.values()))
    if callable(evaluation):
        evaluation = evaluation(args.dataset)
    return pipelines, evaluation


def _load(spec: str) -> Any:
    if not spec.endswith(".py"):
        return importlib.import_module(spec)
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(spec).stem, spec)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _table(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    cells = [[_cell(c, row.get(c)) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def row(values: list[str]) -> str:
        return "│ " + " │ ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)) + " │"

    body = [row(r) for r in cells] or [row([""] * len(columns))]
    return "\n".join(
        [line("┌", "┬", "┐"), row(list(columns)), line("├", "┼", "┤"), *body, line("└", "┴", "┘")]
    )


def _cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column == "deltas":
        return " ".join(f"{k} {v:+.1%}" for k, v in value.items())
    if column == "against":
        return f"{value['verdict']} {value['entry'][:8]}: " + _cell("deltas", value["deltas"])
    if isinstance(value, dict):
        return " ".join(f"{k}={v:.3g}" for k, v in value.items())
    if isinstance(value, str) and len(value) in (16, 26):
        return value[:8]
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
