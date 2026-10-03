"""Pipelines and the evaluation for the Optiver campaign."""

from __future__ import annotations

import importlib.util

from forestry.declare import Evaluation, Pipeline, Schedule, Sealed, Stage, gates, rules
from optiver import steps

ridge = Pipeline(
    name="ridge_3m",
    fit=steps.ridge_fit,
    predict=steps.ridge_predict,
    config=steps.RidgeConfig(train_window_months=3, alpha=1.0),
)
ridge_windows = [ridge.with_config(train_window_months=m).named(f"ridge_{m}m") for m in (1, 3, 6)]

bonsai_depthwise = Pipeline(
    name="bonsai_dw",
    fit=steps.bonsai_fit,
    predict=steps.bonsai_predict,
    config=steps.BonsaiConfig(
        train_window_months=3,
        grower="depthwise",
        max_depth=6,
        n_iters=400,
        learning_rate=0.05,
        max_bin=255,
    ),
    heads=(100, 200, 400),
)
bonsai_leafwise = bonsai_depthwise.with_config(grower="leafwise").named("bonsai_lw")
bonsai_available = importlib.util.find_spec("bonsai") is not None

pipelines = ridge_windows + ([bonsai_depthwise, bonsai_leafwise] if bonsai_available else [])


def evaluation(dataset: str) -> Evaluation:
    """Monthly walk-forward with three ages, a one-month seal and a three-stage funnel."""
    return Evaluation(
        dataset=dataset,
        schedule=Schedule(
            first_cutoff="2021-05-04",
            every_months=1,
            eval_months=1,
            ages=(1, 2, 3),
            embargo_seconds=60,
        ),
        min_folds=3,
        sealed=Sealed(months=1, pass_rule="fold_range", fail="no_deploy"),
        stages=(
            Stage(scorer=steps.fit_metrics, gate=gates.human),
            Stage(
                scorer=steps.taker_sim,
                config=steps.SimConfig(cost_bps=0.5, fidelity="quick"),
                grid=tuple(steps.Exec(threshold_bps=t) for t in (0.0, 2.0, 5.0)),
                gate=gates.pareto_front(2),
            ),
            Stage(
                scorer=steps.taker_sim,
                config=steps.SimConfig(cost_bps=0.5, fidelity="full"),
                gate=gates.vs_baseline,
            ),
        ),
        rule=rules.priority("pnl", "sharpe", "max_dd", "turnover", band=0.02),
        compare_age=1,
    )
