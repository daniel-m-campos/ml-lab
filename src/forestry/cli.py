"""``fy``: the command line over the ledger.

A campaign is a declarations module exposing ``pipelines`` (a list of ``Pipeline``) and
``evaluation`` (an ``Evaluation`` or a function of the dataset id returning one). The ledger root
comes from ``--root`` or ``FORESTRY_ROOT`` (default ``.forestry``).

Examples
--------
$ DS=$(fy freeze optiver.capture:freeze 20)
$ fy run optiver.declarations --dataset "$DS"
$ fy board optiver.declarations --dataset "$DS"
$ fy gate optiver.declarations --dataset "$DS" --pending --advance --why "every corr above 0.15"
$ fy seal optiver.declarations --dataset "$DS"
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
from forestry.ledger import Ledger

DEFAULT_ROOT = ".forestry"
TYPE_KEY = "__type__"
BOARD_COLUMNS = ("seq", "id", "pipeline", "head", "exec", "stage", "status", "reason")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=os.environ.get("FORESTRY_ROOT", DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="call module:function(ledger, *args); prints the id")
    freeze.add_argument("function")
    freeze.add_argument("args", nargs="*")

    for name, text in (
        ("run", "fit, predict and score every pipeline under the evaluation"),
        ("board", "every candidate with its stage, status and aggregate metrics"),
        ("history", "the chain of promotions"),
        ("seal", "score the baseline on the sealed window"),
    ):
        p = sub.add_parser(name, help=text)
        _campaign_args(p)

    gate = sub.add_parser("gate", help="resolve a human gate")
    gate.add_argument("target", help="a candidate id, or the declarations module with --pending")
    gate.add_argument("--dataset", help="with --pending: the dataset id")
    gate.add_argument("--pending", action="store_true", help="every pending candidate")
    gate.add_argument("--advance", action="store_true")
    gate.add_argument("--stop", action="store_true")
    gate.add_argument("--why", required=True)

    decide = sub.add_parser("decide")
    decide.add_argument("--kind", required=True, choices=["promote", "reject", "deploy", "retire"])
    decide.add_argument("--candidate", required=True)
    decide.add_argument("--why", required=True)

    why = sub.add_parser("why")
    why.add_argument("candidate")

    args = parser.parse_args(argv)
    ledger = Ledger.open(pathlib.Path(args.root))
    return _dispatch(args, ledger)


# Private Functions ================================================================================


def _campaign_args(parser: argparse.ArgumentParser):
    parser.add_argument("declarations", help="module name or .py path")
    parser.add_argument("--dataset", required=True, help="dataset id from fy freeze")


def _dispatch(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.command == "freeze":
        print(_load(args.function)(ledger, *args.args))
    elif args.command == "run":
        module, evaluation = _campaign(args)
        report = harness.run(ledger, list(module.pipelines), evaluation)
        print(
            f"fits {report.fits_computed}, predictions {report.predictions_computed}, "
            f"candidates {len(report.candidates)}"
        )
    elif args.command == "board":
        _, evaluation = _campaign(args)
        print(_table(review.board(ledger, evaluation)))
    elif args.command == "history":
        _, evaluation = _campaign(args)
        for row in review.history(ledger, evaluation):
            print(f"{row['candidate'][:8]}  {row['verdict']}  {row['why']}")
    elif args.command == "seal":
        _, evaluation = _campaign(args)
        current = review.baseline(ledger, evaluation)
        if current is None:
            print("no baseline to seal")
            return 1
        verdict = harness.seal(ledger, current["candidate"], evaluation)
        print(f"{verdict.kind}  {_metrics(verdict.metrics)}")
    elif args.command == "gate":
        for cand in _gate_targets(args, ledger):
            harness.gate(ledger, cand, advance=args.advance and not args.stop, why=args.why)
            print(cand)
    elif args.command == "decide":
        print(harness.decide(ledger, kind=args.kind, candidate=args.candidate, why=args.why))
    elif args.command == "why":
        print(json.dumps(review.why(ledger, args.candidate), indent=2, default=str))
    return 0


def _gate_targets(args: argparse.Namespace, ledger: Ledger) -> list[str]:
    if not args.pending:
        return [args.target]
    if args.dataset is None:
        raise SystemExit("fy gate --pending needs the declarations module and --dataset")
    args.declarations = args.target
    _, evaluation = _campaign(args)
    return [r["id"] for r in review.board(ledger, evaluation) if r["status"] == "pending"]


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


def _table(rows: list[dict[str, Any]]) -> str:
    cells = [
        [_cell(row.get(c)) for c in BOARD_COLUMNS] + [_metrics(row["metrics"])] for row in rows
    ]
    header = [*BOARD_COLUMNS, "metrics"]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in value.items()
            if k != TYPE_KEY
        )
    if isinstance(value, str) and len(value) == 16:
        return value[:8]
    return str(value)


def _metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3g}" for k, v in metrics.items())


if __name__ == "__main__":
    raise SystemExit(main())
