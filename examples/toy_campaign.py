"""One campaign on synthetic data, end to end, printing the ledger as it goes.

Run: ``python examples/toy_campaign.py`` (writes to ./.forestry-toy).
"""

from __future__ import annotations

import json
import pathlib
import shutil

from forestry import data, harness, review, toy
from forestry.declare import Evaluation, Pipeline, Schedule, Sealed, Stage, gates
from forestry.ledger import Ledger

ROOT = pathlib.Path(".forestry-toy")


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    ledger = Ledger.open(ROOT)

    frame = toy.generate(
        start="2025-01-01", months=12, rows_per_day=40, seed=7, drift_at="2025-07-01"
    )
    capture = data.freeze_capture(
        ledger, frame, process="toy", params={"seed": 7}, instrument="TOY"
    )
    dataset = data.freeze_dataset(ledger, capture, filters=(), targets=(toy.TARGET,))
    print("dataset", dataset)

    evaluation = Evaluation(
        dataset=dataset,
        schedule=Schedule(first_cutoff="2025-04-01", ages=(1, 2, 3), embargo_seconds=3600),
        sealed=Sealed(months=1),
        stages=(
            Stage(scorer=toy.fit_metrics, gate=gates.human),
            Stage(
                scorer=toy.sign_sim,
                config=toy.SimConfig(cost=0.001, fidelity="quick"),
                grid=tuple(toy.Exec(threshold=t) for t in (0.0, 0.25, 0.5)),
                gate=gates.pareto_front(2),
            ),
            Stage(
                scorer=toy.sign_sim,
                config=toy.SimConfig(cost=0.001, fidelity="full"),
                gate=gates.vs_baseline,
            ),
        ),
    )
    print("evaluation", evaluation.id)

    base = Pipeline(
        name="ridge_1y",
        fit=toy.ridge_fit,
        predict=toy.ridge_predict,
        config=toy.RidgeConfig(
            features=toy.FEATURES, target=toy.TARGET, train_window_months=12, alpha=1.0
        ),
    )
    grid = [base.with_config(train_window_months=m).named(f"ridge_{m}m") for m in (12, 6, 3)]

    report = harness.run(ledger, grid, evaluation)
    print(f"fits {report.fits_computed}, predictions {report.predictions_computed}")
    _show("board after stage 1 (human gate)", review.board(ledger, evaluation))

    for row in review.board(ledger, evaluation):
        ok = row["metrics"]["corr"] > 0.05
        harness.gate(
            ledger,
            row["id"],
            advance=ok,
            why="corr holds" if ok else "corr too low after the drift",
        )

    harness.run(ledger, grid, evaluation)
    _show("board after the funnel", review.board(ledger, evaluation))
    _show("history", review.history(ledger, evaluation))

    for row in review.board(ledger, evaluation):
        if row["status"] == "pending" and row["stage"] == 2:
            harness.decide(
                ledger, kind="promote", candidate=row["id"], why="higher pnl, drawdown acceptable"
            )
            break

    best = review.baseline(ledger, evaluation)["candidate"]
    verdict = harness.seal(ledger, best, evaluation)
    print("seal", verdict.kind, json.dumps(verdict.metrics))
    harness.decide(ledger, kind="deploy", candidate=best, why="campaign done")
    deployment = harness.deploy(ledger, best, env="prod-toy")
    print("deployed bundle", deployment["bundle"][:16])
    _show("history", review.history(ledger, evaluation))


def _show(title: str, rows: list[dict]):
    print(f"\n== {title}")
    for row in rows:
        metrics = {k: round(v, 4) for k, v in row.get("metrics", {}).items()}
        head = {
            k: row[k]
            for k in (
                "pipeline",
                "head",
                "exec",
                "stage",
                "status",
                "reason",
                "kind",
                "why",
                "verdict",
            )
            if k in row
        }
        print(" ", json.dumps({**head, **({"metrics": metrics} if metrics else {})}, default=str))


if __name__ == "__main__":
    main()
