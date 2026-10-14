"""Weak supervision for real-valued labels."""

from .label_model import ContinuousLabelModel, LabelModel
from .synthetic import generate_lfs, generate_synthetic_xy

__all__ = ["ContinuousLabelModel", "LabelModel", "generate_lfs", "generate_synthetic_xy"]
