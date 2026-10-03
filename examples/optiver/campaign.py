"""One Optiver campaign: freeze, run the ridge windows and bonsai, gate, funnel, seal.

Run from the repo root: ``python examples/optiver/campaign.py --stocks 20``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from forestry import harness, review  # noqa: E402
from forestry.ledger import Ledger  # noqa: E402
from optiver import capture, declarations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=20, help="first N stock ids; 0 for all")
    parser.add_argument("--root", default=".forestry-optiver")
    parser.add_argument("--min-corr", type=float, default=0.02, help="stage-1 gate stand-in")
    parser.add_argument("--no-bonsai", action="store_true")
    args = parser.parse_args()
    if not capture.available():
        print(f"missing {capture.TRAIN_CSV}; see examples/optiver/__init__.py")
        return 1

    ledger = Ledger.open(args.root)
    started = time.perf_counter()
    dataset = capture.freeze(ledger, tuple(range(args.stocks)) if args.stocks else None)
    rows = ledger.get("dataset", dataset)["rows"]
    print(f"dataset {dataset} rows={rows} in {time.perf_counter() - started:.1f}s")
    evaluation = declarations.evaluation(dataset)
    pipelines = list(declarations.ridge_windows)
    if declarations.bonsai_available and not args.no_bonsai:
        pipelines += [declarations.bonsai_depthwise, declarations.bonsai_leafwise]

    started = time.perf_counter()
    report = harness.run(ledger, pipelines, evaluation)
    elapsed = time.perf_counter() - started
    print(
        f"stage 1: fits {report.fits_computed}, preds {report.predictions_computed}, {elapsed:.1f}s"
    )
    _show("board after stage 1", review.board(ledger, evaluation))

    for row in review.board(ledger, evaluation):
        if row["status"] != "pending" or row["stage"] != 0:
            continue
        corr = row["metrics"]["corr"]
        harness.gate(ledger, row["id"], advance=corr >= args.min_corr, why=f"corr {corr:.3f}")

    started = time.perf_counter()
    harness.run(ledger, pipelines, evaluation)
    print(f"funnel in {time.perf_counter() - started:.1f}s")
    _show("board after the funnel", review.board(ledger, evaluation))
    _show("history", review.history(ledger, evaluation))

    current = review.baseline(ledger, evaluation)
    if current is not None:
        try:
            verdict = harness.seal(ledger, current["candidate"], evaluation)
            metrics = {k: round(v, 3) for k, v in verdict.metrics.items()}
            print("seal", verdict.kind, json.dumps(metrics))
        except harness.Refused as refused:
            print("seal:", refused)
    pending = sum(r["status"] == "pending" for r in review.board(ledger, evaluation))
    print(f"\nledger at {args.root}; pending decisions: {pending}")
    return 0


def _show(title: str, rows: list[dict]):
    print(f"\n== {title}")
    for row in rows:
        metrics = {k: round(v, 3) for k, v in row.get("metrics", {}).items()}
        exec = row.get("exec")
        head = {
            "pipeline": row.get("pipeline"),
            "head": row.get("head"),
            "thr": exec.get("threshold_bps") if isinstance(exec, dict) else None,
            "stage": row.get("stage"),
            "status": row.get("status") or row.get("kind"),
            "reason": row.get("reason") or row.get("why"),
        }
        shown = {k: v for k, v in head.items() if v is not None}
        if metrics:
            shown["metrics"] = metrics
        print(" ", json.dumps(shown, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
