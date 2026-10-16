"""Regression labeling functions built from tabular features."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import AbstractRegressionLF


class ContinuousLF(AbstractRegressionLF):
    """Weak label from a numeric feature: scaled so its 98th percentile maps to ``y_max``, clipped to
    ``[y_min, y_max]``; zeros (missing) are replaced by ``default_val``."""

    def __init__(self, feature_names):
        self.feature_names = [feature_names] if isinstance(feature_names, str) else list(feature_names)

    def apply(self, df: pd.DataFrame, logarithmic: bool = False, default_val: float = 6, y_min: float = 0,
              y_max: float = 10) -> pd.DataFrame:
        L = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            if col not in df.columns:
                warnings.warn(f"{col} does not exist in df", stacklevel=2)
                continue
            x = df[col].astype(float)
            if logarithmic:
                x = np.log(x + 1)
            q = x.quantile(0.98)
            if not np.isfinite(q) or q == 0:
                warnings.warn(f"98th percentile of {col} is {q}; using its max for scaling", stacklevel=2)
                q = x.abs().max() or 1.0
            weak = np.clip(x / q * y_max, y_min, y_max)
            L[col] = weak.where(weak != 0, default_val)
        return L


class DiscreteLF(AbstractRegressionLF):
    """Weak label from a categorical feature: the mean of ``label_feature`` within each category."""

    def __init__(self, feature_names, label_feature):
        self.feature_names = [feature_names] if isinstance(feature_names, str) else list(feature_names)
        self.label_feature = label_feature

    def apply(self, df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
        L = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            if col not in df.columns:
                warnings.warn(f"{col} does not exist in df", stacklevel=2)
                continue
            group_mean = df.groupby(col)[self.label_feature].mean()
            if debug:
                print(group_mean)
            L[col] = df[col].map(group_mean).to_numpy()
        return L
