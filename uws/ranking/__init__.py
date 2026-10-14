"""Weak supervision for full and partial rankings."""

from .aggregation import kemeny, kemeny_medoid, pairwise_majority, position_estimation, preference_matrix
from .digraph import Edge, RankingDiGraph
from .label_model import InferenceRule, RankingLabelModel, TrainMethod, WeakSupRanking, as_rankings
from .mallows import Mallows
from .ranking import Ranking, RankingUtils, perm2ranking, ranking_to_score
from .synthetic import apply_masks, generate_synthetic_LFs, generate_true_rankings, get_mask, sample_mallows_LFs

__all__ = [
    "Edge", "InferenceRule", "Mallows", "Ranking", "RankingDiGraph", "RankingLabelModel", "RankingUtils",
    "TrainMethod", "WeakSupRanking", "apply_masks", "as_rankings", "generate_synthetic_LFs",
    "generate_true_rankings", "get_mask", "kemeny", "kemeny_medoid", "pairwise_majority", "perm2ranking",
    "position_estimation", "preference_matrix", "ranking_to_score", "sample_mallows_LFs",
]
