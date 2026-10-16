"""Rankings: the :class:`Ranking` type, Kendall tau distances, Mallows models and rank aggregation."""

from .aggregation import kemeny, kemeny_medoid, pairwise_majority, position_estimation, preference_matrix
from .digraph import Edge, RankingDiGraph
from .mallows import Mallows
from .ranking import Ranking, RankingUtils, as_rankings, perm2ranking, ranking_to_score

__all__ = [
    "Edge", "Mallows", "Ranking", "RankingDiGraph", "RankingUtils", "as_rankings", "kemeny", "kemeny_medoid",
    "pairwise_majority", "perm2ranking", "position_estimation", "preference_matrix", "ranking_to_score",
]
