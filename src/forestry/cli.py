"""``fy``: the command line over the log.

A campaign is a declarations module exposing ``pipelines`` (a list of ``Pipeline``) and
``evaluation`` (an ``Evaluation`` or a function of the dataset id returning one). Every campaign
command takes the module and ``--dataset``. The ledger root comes from ``--root`` or
``FORESTRY_ROOT`` (default ``.forestry``); the actor from ``FORESTRY_ACTOR``. Pipelines are named
by name or id prefix.

Examples
--------
$ DS=$(fy freeze optiver.capture:freeze 20)
$ C="optiver.declarations --dataset $DS"
$ fy run $C
$ fy decide $C ridge_3m --kind promote --why "the model in production"
$ fy board $C
$ fy decide $C bonsai_lw --kind promote --why "pnl +7.6%, drawdown accepted"
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

from forestry import harness, review
from forestry.declare import Evaluation
from forestry.ledger import Event, Ledger

DEFAULT_ROOT = ".forestry"
BOARD_COLUMNS = ("entry", "pipeline", "status", "verdict", "deltas", "metrics", "why")
HISTORY_COLUMNS = ("id", "pipeline", "key", "against", "why", "actor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=os.environ.get("FORESTRY_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze", help="call module:function(ledger, *args); prints the id")
    p.add_argument("function", help="module:function(ledger, *args)")
    p.add_argument("args", nargs="*")
    p.set_defaults(handler=_freeze)

    p = sub.add_parser("run", help="fit, predict and score every pipeline")
    _campaign_arguments(p)
    p.set_defaults(handler=_run)

    p = sub.add_parser("board", help="every pipeline against the baseline; one name for detail")
    _campaign_arguments(p)
    p.add_argument("pipeline", nargs="?", help="one pipeline's detail as JSON")
    p.set_defaults(handler=_board)

    p = sub.add_parser("decide", help="record promote or reject on a pipeline")
    _campaign_arguments(p)
    p.add_argument("pipeline")
    p.add_argument("--kind", required=True, choices=["promote", "reject"])
    p.add_argument("--why", required=True)
    p.set_defaults(handler=_decide)

    p = sub.add_parser("history", help="the chain of promotions")
    _campaign_arguments(p)
    p.set_defaults(handler=_history)

    args = parser.parse_args(argv)
    ledger = Ledger.open(pathlib.Path(args.root))
    try:
        return args.handler(args, ledger)
    except (harness.Refused, KeyError) as refused:
        print(f"fy {args.command}: {refused}", file=sys.stderr)
        return 1


# Handlers =========================================================================================


def _campaign_arguments(p: argparse.ArgumentParser):
    p.add_argument("declarations", help="module name or .py path")
    p.add_argument("--dataset", required=True, help="dataset id from fy freeze")


def _freeze(args: argparse.Namespace, ledger: Ledger) -> int:
    print(_load(args.function)(ledger, *args.args))
    return 0


def _run(args: argparse.Namespace, ledger: Ledger) -> int:
    module, evaluation = _campaign(args)
    report = harness.run(ledger, list(module.pipelines), evaluation)
    print(
        f"run {report.run}: fits {report.fits_computed}, predictions "
        f"{report.predictions_computed}, entries {report.entries_scored}"
    )
    return 0


def _board(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _campaign(args)
    if args.pipeline:
        pipeline = _resolve(ledger, evaluation, args.pipeline)
        print(json.dumps(review.detail(ledger, evaluation, pipeline), indent=2, default=str))
    else:
        print(_table(BOARD_COLUMNS, review.board(ledger, evaluation)))
    return 0


def _decide(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _campaign(args)
    pipeline = _resolve(ledger, evaluation, args.pipeline)
    print(harness.decide(ledger, pipeline, evaluation, kind=args.kind, why=args.why))
    return 0


def _history(args: argparse.Namespace, ledger: Ledger) -> int:
    _, evaluation = _campaign(args)
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
        raise harness.Refused(f"pipeline {name!r}: {len(matches)} matches under this evaluation")
    return matches[0]


def _campaign(args: argparse.Namespace) -> tuple[Any, Evaluation]:
    module = _load(args.declarations)
    evaluation = module.evaluation
    if callable(evaluation):
        evaluation = evaluation(args.dataset)
    return module, evaluation


def _load(spec: str) -> Any:
    target, _, name = spec.partition(":")
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(pathlib.Path(target).stem, target)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name) if name else module


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
