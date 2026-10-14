"""Universalizing Weak Supervision (Shin et al., ICLR 2022): label models for any label type."""

from .base import BaseLabelModel
from .config import InferenceRule, RankingConfig, TrainMethod
from .metric import MetricLabelModel
from .ranking import Mallows, Ranking, RankingLabelModel, RankingUtils
from .regression import ContinuousLabelModel
from .version import __version__

__all__ = [
    "BaseLabelModel", "ContinuousLabelModel", "InferenceRule", "Mallows", "MetricLabelModel", "Ranking",
    "RankingConfig", "RankingLabelModel", "RankingUtils", "TrainMethod", "__version__",
]


def __getattr__(name):  # lazy optional imports
    if name == "GeodesicLabelModel":
        from .geodesic import GeodesicLabelModel

        return GeodesicLabelModel
    if name == "BinaryLabelModel":
        from .binary import BinaryLabelModel

        return BinaryLabelModel
    raise AttributeError(f"module 'uws' has no attribute {name!r}")
