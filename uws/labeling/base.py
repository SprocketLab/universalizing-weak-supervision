"""Abstract labeling functions."""

from abc import ABC, abstractmethod


class AbstractRankingLF(ABC):
    """``apply(lst_feature_map)``: a list of per-item feature dicts -> :class:`uws.ranking.Ranking`."""

    @abstractmethod
    def apply(self, lst_feature_map):
        ...


class AbstractRegressionLF(ABC):
    """``apply(df)``: a DataFrame -> a DataFrame of weak labels (one column per source feature)."""

    @abstractmethod
    def apply(self, df):
        ...
