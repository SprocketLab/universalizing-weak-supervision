"""Universalizing Weak Supervision (Shin et al., ICLR 2022): weak supervision for any label type.

* :mod:`uws.label_model`: label models for rankings, real values, finite metric spaces, manifolds and binary labels.
* :mod:`uws.end_model`: end models trained on the resulting labels.
* :mod:`uws.ranking`: rankings, Kendall tau distances, Mallows models and rank aggregation.
* :mod:`uws.synthetic`: synthetic labels and labeling functions.
* :mod:`uws.labeling`, :mod:`uws.datasets`: labeling functions and the datasets used in the paper.
"""

from .version import __version__

__all__ = ["__version__"]
