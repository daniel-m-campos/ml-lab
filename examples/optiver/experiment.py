"""Pipelines and the evaluation for the Optiver example; ``fy`` loads this module."""

from __future__ import annotations

from forestry.experiment import Evaluation, Pipeline
from forestry.splits import CalendarWalkForward
from optiver import steps

ridge = Pipeline(
    name="ridge_3m",
    fit=steps.ridge_fit,
    predict=steps.ridge_predict,
    save=steps.ridge_save,
    load=steps.ridge_load,
    config=steps.RidgeConfig(train_window_months=3, alpha=1.0),
)
ridge_windows = [
    ridge.with_config(train_window_months=m).named(f"ridge_{m}m") for m in (1, 3, 6)
]

bonsai_depthwise = Pipeline(
    name="bonsai_dw",
    fit=steps.bonsai_fit,
    predict=steps.bonsai_predict,
    save=steps.bonsai_save,
    load=steps.bonsai_load,
    config=steps.BonsaiConfig(
        train_window_months=3,
        grower="depthwise",
        max_depth=6,
        n_iters=200,
        learning_rate=0.05,
        max_bin=255,
    ),
)
bonsai_leafwise = bonsai_depthwise.with_config(grower="leafwise").named("bonsai_lw")
bonsai_available = steps.bonsai is not None

pipelines = ridge_windows + (
    [bonsai_depthwise, bonsai_leafwise] if bonsai_available else []
)


def evaluation(dataset: str) -> Evaluation:
    """Monthly walk-forward with three horizons, scored by the taker simulation."""
    return Evaluation(
        dataset=dataset,
        split=CalendarWalkForward(
            first_cutoff="2021-05-04",
            every_months=1,
            eval_months=1,
            horizons=(1, 2, 3),
            embargo_seconds=60,
            min_folds=3,
        ),
        scorer=steps.taker_sim,
        config=steps.SimConfig(cost_bps=0.5, threshold_bps=0.0),
    )
