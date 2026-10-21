"""ml-lab: a local-first ledger of experimentation on frozen datasets."""

from __future__ import annotations

import importlib.metadata

from ml_lab.analytics import paired, paired_sets, predictions
from ml_lab.dataset import Dataset, FitView, Range, Segments, load, record
from ml_lab.experiment import Evaluation, Fitted, Pipeline, Scorer, mean_blend
from ml_lab.ledger import Ledger, note
from ml_lab.splits import Fold, inner_split

__version__ = importlib.metadata.version("ml-lab")
__all__ = [
    # declaring a study: the ledger, its datasets, pipelines and evaluations
    "Ledger",
    "Dataset",
    "record",
    "Pipeline",
    "Scorer",
    "Evaluation",
    # writing a pipeline's functions
    "FitView",
    "Fitted",
    "Fold",
    "Range",
    "Segments",
    "inner_split",
    "mean_blend",
    # reading the log back
    "load",
    "predictions",
    "paired",
    "paired_sets",
    "note",
]
