"""The universal label model: weak supervision over any finite metric space."""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .base import BaseLabelModel
from .triplets import individual_distances


class MetricLabelModel(BaseLabelModel):
    """Label model for labels living in a finite metric space ``{0, .., k-1}`` with distance matrix ``D``.

    LF ``j`` is modeled as an exponential family ``P(lambda_j = x | y) ∝ exp(-theta_j d(x, y)^q)``.
    :meth:`fit` estimates each LF's expected distance to the latent label from LF-LF distances with
    triplets (no labels needed) and sets ``theta_j = 1 / E[d(lambda_j, y)^power]`` (or the exact
    exponential-family inversion with ``exact_theta=True``). :meth:`predict` returns the weighted Fréchet
    mean ``argmin_y sum_j theta_j d(lambda_j, y)^center_power`` over the candidate labels.

    Parameters
    ----------
    D : (k, k) array or callable
        Distance matrix over the label space, or a function ``d(x, y)`` together with ``candidates``.
    candidates : sequence, optional
        Label space when ``D`` is a callable (default: the distinct LF outputs).
    power : float
        Exponent of the distance used for parameter estimation.
    center_power : float
        Exponent of the distance in the Fréchet mean (2 = mean, 1 = median).
    method : str
        Triplet estimator (see :func:`uws.label_model.triplets.individual_distances`).
    exact_theta : bool
        Invert ``E_theta[d^power] = e`` under the exponential family, averaged over centers (finite ``D``).
    abstain : int
        LF output meaning "abstain" (ignored in both fitting and prediction).
    """

    def __init__(self, D: Union[np.ndarray, Callable], candidates: Optional[Sequence] = None,
                 power: float = 1.0, center_power: float = 2.0, method: str = "median_triplet",
                 exact_theta: bool = False, abstain: int = -1):
        self.callable = callable(D)
        self.distance = D if self.callable else None
        self.D = None if self.callable else np.asarray(D, dtype=float)
        self.candidates = candidates
        self.power = power
        self.center_power = center_power
        self.method = method
        self.exact_theta = exact_theta
        self.abstain = abstain
        self.thetas = None

    # ------------------------------------------------------------------ helpers
    def _dist(self, x, y):
        return self.distance(x, y) if self.callable else self.D[x, y]

    def lf_distance_matrix(self, L) -> np.ndarray:
        """Mean ``d^power`` between every pair of LFs over examples where both vote."""
        n, m = len(L), len(L[0])
        E = np.zeros((m, m))
        if not self.callable:
            L = np.asarray(L, dtype=int)
            Dp = self.D**self.power
            for a, b in itertools.combinations(range(m), 2):
                ok = (L[:, a] != self.abstain) & (L[:, b] != self.abstain)
                if not ok.any():
                    raise ValueError(f"LFs {a} and {b} never vote on the same example")
                E[a, b] = E[b, a] = Dp[L[ok, a], L[ok, b]].mean()
        else:
            for a, b in itertools.combinations(range(m), 2):
                vals = [self._dist(L[i][a], L[i][b]) ** self.power for i in range(n)
                        if L[i][a] != self.abstain and L[i][b] != self.abstain]
                E[a, b] = E[b, a] = np.mean(vals)
        return E

    def _theta_from_expected(self, e: np.ndarray) -> np.ndarray:
        if not self.exact_theta:
            return 1.0 / e
        if self.callable:
            raise ValueError("exact_theta needs a distance matrix")
        from scipy.optimize import brentq

        Dp = self.D**self.power

        def mean_dist(theta):  # E_theta[d^p] averaged over centers
            P = np.exp(-theta * Dp)
            P /= P.sum(1, keepdims=True)
            return float((P * Dp).sum(1).mean())

        out = []
        for ej in e:
            lo, hi = 1e-6, 1e3
            if mean_dist(lo) <= ej:
                out.append(lo)
            elif mean_dist(hi) >= ej:
                out.append(hi)
            else:
                out.append(brentq(lambda t, ej=ej: mean_dist(t) - ej, lo, hi))
        return np.array(out)

    # ------------------------------------------------------------------ API
    def fit(self, L, groups: Optional[Sequence] = None) -> MetricLabelModel:
        """Estimate per-LF parameters from unlabeled LF outputs ``L`` (n x m)."""
        self.E_ = self.lf_distance_matrix(L)
        self.expected_dists = individual_distances(self.E_, self.method, groups=groups)
        self.thetas = self._theta_from_expected(self.expected_dists)
        return self

    def predict_costs(self, L, weights=None) -> np.ndarray:
        """``(n, k)`` weighted Fréchet costs of every candidate label (finite ``D`` only)."""
        if self.callable:
            raise ValueError("predict_costs needs a distance matrix")
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        w = self.get_weights() if weights is None else np.asarray(weights, dtype=float)
        k = self.D.shape[0]
        V = np.zeros((n, k))
        for j in range(m):
            ok = L[:, j] != self.abstain
            np.add.at(V, (np.nonzero(ok)[0], L[ok, j]), w[j])
        return V @ (self.D**self.center_power)

    def predict_proba(self, L, temperature: float = 1.0, weights=None) -> np.ndarray:
        """Softmax over negative Fréchet costs (a Gibbs posterior over the label space)."""
        c = -self.predict_costs(L, weights) / temperature
        c -= c.max(1, keepdims=True)
        p = np.exp(c)
        return p / p.sum(1, keepdims=True)

    def predict(self, L, weights=None, vote: bool = False):
        """Weighted Fréchet mean of the LF outputs for every example (``vote=True``: unit weights)."""
        m = len(L[0])
        w = np.ones(m) if vote else (self.get_weights() if weights is None else np.asarray(weights))
        if not self.callable:
            return np.argmin(self.predict_costs(L, w), axis=1)
        cands = self.candidates
        if cands is None:
            cands = list({x for row in L for x in row if x != self.abstain})
        out = []
        for row in L:
            costs = [sum(w[j] * self._dist(row[j], c) ** self.center_power for j in range(m)
                         if row[j] != self.abstain) for c in cands]
            out.append(cands[int(np.argmin(costs))])
        return out

    def score(self, L, Y) -> dict:
        Y_hat = self.predict(L)
        acc = float(np.mean([a == b for a, b in zip(Y_hat, Y)]))
        md = float(np.mean([self._dist(a, b) for a, b in zip(Y_hat, Y)]))
        return {"accuracy": acc, "mean_distance": md}
