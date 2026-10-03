"""``fy``: the command line over the ledger. Declarations are loaded as ``module.py:object``.

Examples
--------
$ fy run -p pipelines/ridge.py:grid -e evaluations/es_q3.py:es_q3
$ fy board -e evaluations/es_q3.py:es_q3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

from forestry import harness, review
from forestry.ledger import Ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fy")
    parser.add_argument("--root", default=".forestry", help="ledger directory")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit, predict and score pipelines under an evaluation")
    run.add_argument("-p", "--pipelines", required=True, help="module.py:object (Pipeline or list)")
    run.add_argument("-e", "--evaluation", required=True)

    for name in ("board", "history"):
        p = sub.add_parser(name)
        p.add_argument("-e", "--evaluation", required=True)

    gate = sub.add_parser("gate", help="resolve a human gate")
    gate.add_argument("candidate")
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


def _dispatch(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.command == "run":
        pipelines = _load(args.pipelines)
        report = harness.run(
            ledger,
            list(pipelines) if isinstance(pipelines, (list, tuple)) else [pipelines],
            _load(args.evaluation),
        )
        _emit(
            {
                "fits_computed": report.fits_computed,
                "predictions_computed": report.predictions_computed,
                "candidates": report.candidates,
            }
        )
    elif args.command == "board":
        _emit(review.board(ledger, _load(args.evaluation)))
    elif args.command == "history":
        _emit(review.history(ledger, _load(args.evaluation)))
    elif args.command == "gate":
        _emit(
            harness.gate(
                ledger, args.candidate, advance=args.advance and not args.stop, why=args.why
            )
        )
    elif args.command == "decide":
        _emit(harness.decide(ledger, kind=args.kind, candidate=args.candidate, why=args.why))
    elif args.command == "why":
        _emit(review.why(ledger, args.candidate))
    return 0


def _load(spec: str) -> Any:
    path, _, name = spec.partition(":")
    module_spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def _emit(obj: Any):
    print(json.dumps(obj, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
