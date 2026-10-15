"""End models trained on the labels a label model produces.

* :class:`RankingEndModel`: a feed-forward ranker trained with the ListMLE loss (needs ``uws[l2r]``).
* :class:`GeodesicRegressor`: geodesic regression on a Riemannian manifold (needs ``uws[geodesic]``).
"""

from .geodesic import GeodesicRegressor
from .ranking import RankingEndModel

__all__ = ["GeodesicRegressor", "RankingEndModel"]

