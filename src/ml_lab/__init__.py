"""ml-lab: a local-first ledger of experimentation on frozen datasets."""

from __future__ import annotations

from ml_lab.analytics import paired
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Ledger
from ml_lab.panel import Panel
from ml_lab.splits import BlockedKFold, CalendarWalkForward, Holdout, WalkForward

__version__ = "0.0.1"
__all__ = [
    "Panel",
    "BlockedKFold",
    "CalendarWalkForward",
    "Evaluation",
    "Holdout",
    "Ledger",
    "Pipeline",
    "Scorer",
    "WalkForward",
    "paired",
]
