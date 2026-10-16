"""Label models: estimate labeling-function accuracies without labels and aggregate LF outputs.

Every model follows the same interface (:class:`BaseLabelModel`): ``fit(L)``, ``predict(L)``,
``get_weights()``, ``save()`` / ``load()``.

* :class:`RankingLabelModel`: full and partial rankings (Mallows LFs).
* :class:`ContinuousLabelModel`: real-valued labels (Gaussian LFs).
* :class:`MetricLabelModel`: labels in any finite metric space.
* :class:`GeodesicLabelModel`: labels on a Riemannian manifold (needs ``universalizing-weak-supervision[geodesic]``).
* :func:`BinaryLabelModel`: binary labels through FlyingSquid (needs ``universalizing-weak-supervision[binary]``).
"""

from ..config import InferenceRule, RankingConfig, TrainMethod
from .base import BaseLabelModel
from .binary import BinaryLabelModel
from .geodesic import GeodesicLabelModel
from .metric import MetricLabelModel
from .ranking import RankingLabelModel, WeakSupRanking
from .regression import ContinuousLabelModel, LabelModel
from .triplets import individual_distances, rank_one_fit

__all__ = [
    "BaseLabelModel", "BinaryLabelModel", "ContinuousLabelModel", "GeodesicLabelModel", "InferenceRule",
    "LabelModel", "MetricLabelModel", "RankingConfig", "RankingLabelModel", "TrainMethod", "WeakSupRanking",
    "individual_distances", "rank_one_fit",
]

