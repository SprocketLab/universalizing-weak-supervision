"""Ranking labeling functions."""

from __future__ import annotations

from ..ranking.ranking import Ranking, RankingUtils
from .base import AbstractRankingLF


class FeatureRankingLF(AbstractRankingLF):
    """Rank items by one feature (``highest_first``: descending)."""

    def __init__(self, rank_on_feature, d: int, highest_first: bool = True, r_utils: RankingUtils = None):
        self.rank_on_feature = rank_on_feature
        self.highest_first = highest_first
        self.r_utils = RankingUtils(d) if r_utils is None else r_utils

    def __repr__(self):
        return f"FeatureRankingLF({self.rank_on_feature!r}, highest_first={self.highest_first})"

    def _rank(self, values) -> Ranking:
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=self.highest_first)
        return Ranking(order, self.r_utils)

    def apply(self, lst_feature_map) -> Ranking:
        return self._rank([fm[self.rank_on_feature] for fm in lst_feature_map])

    def apply_mat(self, X):
        """``X``: (n, d, num_features) array, ``rank_on_feature`` a feature index."""
        return [self._rank([X[row][i][self.rank_on_feature] for i in range(X.shape[1])])
                for row in range(X.shape[0])]
