"""``fy``: the command line over the ledger.

A campaign is a declarations module exposing ``pipelines`` (a list of ``Pipeline``) and
``evaluation`` (an ``Evaluation`` or a function of the dataset id returning one). Every campaign
command takes the module and ``--dataset``. The ledger root comes from ``--root`` or
``FORESTRY_ROOT`` (default ``.forestry``). Candidates are named by id, id prefix or pipeline name.

Examples
--------
$ DS=$(fy freeze optiver.capture:freeze 20)
$ C="optiver.declarations --dataset $DS"
$ fy run $C
$ fy board $C
$ fy decide $C ridge_3m --kind promote --why "the model in production"
$ fy compare $C
$ fy decide $C bonsai_lw --kind promote --why "pnl +2.4%, drawdown within tolerance"
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
BOARD_COLUMNS = ("seq", "id", "pipeline", "status", "verdict", "metrics", "reason")
COMPARE_COLUMNS = ("id", "pipeline", "verdict", "deltas", "metrics")
HISTORY_COLUMNS = ("decision", "candidate", "pipeline", "verdict", "deltas", "why")
COMMANDS = {
    "run": "fit, predict and score every pipeline under the evaluation",
    "board": "every candidate with its status and aggregate metrics",
    "compare": "write a comparison of each named candidate (default: all) to the baseline",
    "decide": "record promote or reject on a candidate",
    "history": "the chain of promotions",
    "why": "one candidate: config against the baseline, folds, comparisons, decisions",
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
        if name == "compare":
            p.add_argument("candidates", nargs="*")
        if name in ("decide", "why"):
            p.add_argument("candidate")
        if name == "decide":
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
            f"candidates {len(report.candidates)}"
        )
    elif args.command == "board":
        print(_table(BOARD_COLUMNS, review.board(ledger, evaluation)))
    elif args.command == "compare":
        rows = [
            harness.compare(ledger, cid, evaluation)
            for cid in _compare_targets(args, ledger, evaluation)
        ]
        for row in rows:
            row["id"] = row["challenger"]
            row["pipeline"] = _pipeline_name(ledger, row["challenger"])
            row["metrics"] = row["challenger_metrics"]
        print(_table(COMPARE_COLUMNS, rows))
    elif args.command == "decide":
        candidate = _resolve(ledger, evaluation, args.candidate)
        print(harness.decide(ledger, candidate, kind=args.kind, why=args.why))
    elif args.command == "history":
        rows = [{**r, "decision": r["id"]} for r in review.history(ledger, evaluation)]
        print(_table(HISTORY_COLUMNS, rows))
    elif args.command == "why":
        candidate = _resolve(ledger, evaluation, args.candidate)
        print(json.dumps(review.why(ledger, candidate), indent=2, default=str))
    return 0


def _compare_targets(args: argparse.Namespace, ledger: Ledger, evaluation: Evaluation) -> list[str]:
    if args.candidates:
        return [_resolve(ledger, evaluation, c) for c in args.candidates]
    return [
        c["id"]
        for c in ledger.where(Kinds.CANDIDATE, evaluation=evaluation.id)
        if c["status"] == harness.Status.SCORED
    ]


def _resolve(ledger: Ledger, evaluation: Evaluation, name: str) -> str:
    """A candidate id from an id, an id prefix or a pipeline name under the evaluation."""
    candidates = ledger.where(Kinds.CANDIDATE, evaluation=evaluation.id)
    matches = [c["id"] for c in candidates if c["id"].startswith(name)]
    if not matches:
        matches = [c["id"] for c in candidates if _pipeline_name(ledger, c["id"]) == name]
    if len(matches) != 1:
        raise harness.Refused(f"candidate {name!r}: {len(matches)} matches")
    return matches[0]


def _pipeline_name(ledger: Ledger, candidate_id: str) -> str:
    cand = ledger.get(Kinds.CANDIDATE, candidate_id)
    return ledger.get(Kinds.PIPELINE, cand["pipeline"])["name"]


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
    if isinstance(value, dict):
        return " ".join(f"{k}={v:.3g}" for k, v in value.items())
    if isinstance(value, str) and len(value) == 16:
        return value[:8]
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
