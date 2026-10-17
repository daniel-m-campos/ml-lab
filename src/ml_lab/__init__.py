"""ml-lab: a local-first ledger of experimentation on frozen datasets."""

from __future__ import annotations

from ml_lab.analytics import paired, paired_sets
from ml_lab.dataset import Dataset, FitView, Range, Segments, load, record
from ml_lab.experiment import Evaluation, Fitted, Pipeline, Scorer, mean_blend
from ml_lab.ledger import Ledger, note
from ml_lab.splits import Fold, inner_split

__version__ = "0.0.1"
__all__ = [
    "Dataset",
    "Evaluation",
    "FitView",
    "Fitted",
    "Fold",
    "Ledger",
    "Pipeline",
    "Range",
    "Scorer",
    "Segments",
    "inner_split",
    "load",
    "mean_blend",
    "note",
    "paired",
    "paired_sets",
    "record",
]
