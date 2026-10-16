"""Synthetic hyperbolic regression data and noisy manifold LFs (from the geodesic regression notebook)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_params(man, dim: int, zero: bool = False):
    """Random intercept ``p`` on the manifold and unit tangent direction ``beta`` at ``p``."""
    import torch

    p = torch.randn((1, dim)) * 2.0
    if zero:
        p /= 1000
    p = man.projx(p)
    orig = man.origin((1, dim))
    beta = torch.randn((1, dim)) * 2.0
    if zero:
        beta /= 1000
    beta /= beta.norm(dim=-1)
    beta = man.transp(orig, p, man.proju(orig, beta))
    return p, beta


def add_noise(man, y, sig: float = 1.0):
    """Wrapped Gaussian noise: tangent noise at the origin, transported to ``y`` and mapped back."""
    import torch

    n, dim = y.shape
    orig = man.origin((1, dim))
    eps = man.transp(orig, y, man.proju(orig, torch.randn(n, dim) * sig))
    return man.expmap(y, eps)


def generate_data(man, n: int, p, beta, sig: float = 1.0):
    """``x ~ U[-5, 5]``, ``y = exp_p(x beta)`` plus noise. Returns ``x, y_noisy, y``."""
    import torch

    x = (torch.rand(n, 1) * 10.0) - 5.0
    y = man.expmap(p, x * beta)
    return x, add_noise(man, y, sig), y


class LFGenerator:
    """``num_lfs`` noisy copies of the true labels with noise scales drawn from ``[siglb, sigub]``
    (shifted by ``sep`` for a random half when ``heterogeneous``)."""

    def __init__(self, man, num_lfs: int, siglb: float = 2, sigub: float = 5, heterogeneous: bool = True,
                 rng: Optional[np.random.Generator] = None):
        self.man = man
        self.num_lfs = num_lfs
        self.siglb, self.sigub = siglb, sigub
        self.heterogeneous = heterogeneous
        self.rng = np.random.default_rng() if rng is None else rng
        self.sigs: list[float] = []

    def fit(self, sep: float = 15):
        self.sigs = []
        for _ in range(self.num_lfs):
            coin = int(self.rng.integers(0, 2)) * self.heterogeneous
            self.sigs.append(float(self.rng.uniform(self.siglb + coin * sep, self.sigub + coin * sep)))
        return self

    def predict(self, Y_true):
        return [add_noise(self.man, Y_true, sig=s) for s in self.sigs]
