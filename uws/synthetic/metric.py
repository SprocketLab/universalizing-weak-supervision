"""LF outputs for a finite metric space."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_metric_lfs(D: np.ndarray, Y: np.ndarray, thetas, power: float = 1.0,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample LF outputs ``P(x | y) ∝ exp(-theta d(x, y)^power)`` for every true label in ``Y``."""
    rng = np.random.default_rng() if rng is None else rng
    D = np.asarray(D, dtype=float)
    k = D.shape[0]
    L = np.zeros((len(Y), len(thetas)), dtype=int)
    for j, theta in enumerate(thetas):
        P = np.exp(-theta * D**power)
        P /= P.sum(1, keepdims=True)
        cum = P[Y].cumsum(1)
        L[:, j] = np.minimum((rng.random((len(Y), 1)) > cum).sum(1), k - 1)
    return L
