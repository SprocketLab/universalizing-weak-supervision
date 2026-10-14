"""Weak supervision for manifold-valued labels (requires ``uws[geodesic]``)."""

from .label_model import GeodesicLabelModel
from .regressor import GeodesicRegressor
from .synthetic import LFGenerator, add_noise, generate_data, sample_params

__all__ = ["GeodesicLabelModel", "GeodesicRegressor", "LFGenerator", "add_noise", "generate_data", "sample_params"]
