"""ml-lab: a local-first ledger of experimentation on frozen datasets."""

from __future__ import annotations

from ml_lab.analytics import paired
from ml_lab.dataset import Dataset, Range, Segments, load, record
from ml_lab.experiment import Evaluation, Pipeline, Scorer
from ml_lab.ledger import Ledger
from ml_lab.splits import Fold

__version__ = "0.0.1"
__all__ = [
    "Dataset",
    "Evaluation",
    "Fold",
    "Ledger",
    "Pipeline",
    "Range",
    "Scorer",
    "Segments",
    "load",
    "paired",
    "record",
]
