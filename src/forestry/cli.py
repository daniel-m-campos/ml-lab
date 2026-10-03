"""``fy``: the command line over the ledger.

A campaign is a declarations module exposing ``pipelines`` (a list of ``Pipeline``) and
``evaluation`` (an ``Evaluation`` or a function of the dataset id returning one). Every campaign
command takes the module and ``--dataset``. The ledger root comes from ``--root`` or
``FORESTRY_ROOT`` (default ``.forestry``). Pipelines are named by name or id prefix.

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
from forestry.ledger import Kinds, Ledger

DEFAULT_ROOT = ".forestry"
BOARD_COLUMNS = ("id", "pipeline", "status", "verdict", "deltas", "metrics", "why")
HISTORY_COLUMNS = ("id", "pipeline", "pipeline_id", "against", "why")
COMMANDS = {
    "run": "fit, predict and score every pipeline under the evaluation",
    "board": "every scored pipeline against the baseline; one name for its detail",
    "decide": "record promote or reject on a pipeline",
    "history": "the chain of promotions",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=os.environ.get("FORESTRY_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="call module:function(ledger, *args); prints the id")
    freeze.add_argument("function")
    freeze.add_argument("args", nargs="*")
    for name, text in COMMANDS.items():
        p = sub.add_parser(name, help=text)
        p.add_argument("declarations", help="module name or .py path")
        p.add_argument("--dataset", required=True, help="dataset id from fy freeze")
        if name == "board":
            p.add_argument("pipeline", nargs="?")
        if name == "decide":
            p.add_argument("pipeline")
            p.add_argument("--kind", required=True, choices=["promote", "reject"])
            p.add_argument("--why", required=True)

    args = parser.parse_args(argv)
    ledger = Ledger.open(pathlib.Path(args.root))
    try:
        return _dispatch(args, ledger)
    except (harness.Refused, KeyError) as refused:
        print(f"fy {args.command}: {refused}", file=sys.stderr)
        return 1


# Private Functions ================================================================================


def _dispatch(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.command == "freeze":
        print(_load(args.function)(ledger, *args.args))
        return 0
    module, evaluation = _campaign(args)
    if args.command == "run":
        report = harness.run(ledger, list(module.pipelines), evaluation)
        print(
            f"fits {report.fits_computed}, predictions {report.predictions_computed}, "
            f"pipelines {len(report.scored)}"
        )
    elif args.command == "board" and args.pipeline:
        pipeline = _resolve(ledger, evaluation, args.pipeline)
        print(json.dumps(review.detail(ledger, evaluation, pipeline), indent=2, default=str))
    elif args.command == "board":
        print(_table(BOARD_COLUMNS, review.board(ledger, evaluation)))
    elif args.command == "decide":
        pipeline = _resolve(ledger, evaluation, args.pipeline)
        print(harness.decide(ledger, pipeline, evaluation, kind=args.kind, why=args.why))
    elif args.command == "history":
        print(_table(HISTORY_COLUMNS, review.history(ledger, evaluation)))
    return 0


def _resolve(ledger: Ledger, evaluation: Evaluation, name: str) -> str:
    """A pipeline id from its name or an id prefix, among those scored under the evaluation."""
    scored = [s["pipeline"] for s in ledger.where(Kinds.SCORE, evaluation=evaluation.id)]
    matches = [p for p in scored if p.startswith(name)]
    if not matches:
        matches = [p for p in scored if ledger.get(Kinds.PIPELINE, p)["name"] == name]
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
        return f"{value['verdict']} {value['baseline'][:8]}: " + _cell("deltas", value["deltas"])
    if isinstance(value, dict):
        return " ".join(f"{k}={v:.3g}" for k, v in value.items())
    if isinstance(value, str) and len(value) == 16:
        return value[:8]
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
