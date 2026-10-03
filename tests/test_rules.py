"""Ordering rules: Pareto and priority order with a band."""

from __future__ import annotations

from forestry.declare import rules
from forestry.harness import Verdict, _dominance, _ranked

DIRECTIONS = {"pnl": "max", "sharpe": "max", "max_dd": "min", "turnover": "min"}
PRIORITY = rules.priority("pnl", "sharpe", "max_dd", "turnover", band=0.02)


def test_priority_decides_on_the_first_metric_outside_the_band():
    a = {"pnl": 110.0, "sharpe": 1.0, "max_dd": 50.0, "turnover": 10.0}
    b = {"pnl": 100.0, "sharpe": 2.0, "max_dd": 10.0, "turnover": 1.0}
    assert _dominance(a, b, DIRECTIONS, PRIORITY) == Verdict.DOMINATES
    assert _dominance(b, a, DIRECTIONS, PRIORITY) == Verdict.DOMINATED


def test_priority_falls_through_inside_the_band_and_respects_direction():
    a = {"pnl": 101.0, "sharpe": 2.0, "max_dd": 30.0, "turnover": 10.0}
    b = {"pnl": 100.0, "sharpe": 2.0, "max_dd": 20.0, "turnover": 10.0}
    assert _dominance(a, b, DIRECTIONS, PRIORITY) == Verdict.DOMINATED
    tie = {"pnl": 101.0, "sharpe": 2.01, "max_dd": 20.2, "turnover": 10.1}
    assert _dominance(tie, b, DIRECTIONS, PRIORITY) == Verdict.INCOMPARABLE


def test_ranking_orders_by_priority_and_pareto_puts_the_front_first():
    metrics = {
        "x": {"pnl": 100.0, "sharpe": 1.0, "max_dd": 10.0, "turnover": 1.0},
        "y": {"pnl": 120.0, "sharpe": 0.5, "max_dd": 50.0, "turnover": 9.0},
        "z": {"pnl": 90.0, "sharpe": 3.0, "max_dd": 5.0, "turnover": 1.0},
    }
    assert _ranked(metrics, DIRECTIONS, PRIORITY) == ["y", "x", "z"]
    pareto = _ranked(metrics, DIRECTIONS, rules.pareto)
    assert set(pareto[:2]) == {"y", "z"} or set(pareto) == {"x", "y", "z"}
