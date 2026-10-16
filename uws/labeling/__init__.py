"""Labeling functions."""

from .base import AbstractRankingLF, AbstractRegressionLF
from .ranking import FeatureRankingLF
from .regression import ContinuousLF, DiscreteLF

__all__ = ["AbstractRankingLF", "AbstractRegressionLF", "ContinuousLF", "DiscreteLF", "FeatureRankingLF"]
