"""Mallows model over rankings with the Kendall tau distance."""

from __future__ import annotations

import itertools
from collections import defaultdict
from math import exp, log
from typing import Optional

import numpy as np
import scipy.optimize

from .ranking import Ranking, RankingUtils


class Mallows:
    """Mallows model :math:`P(\\pi) \\propto \\exp(-\\theta\\, d_\\tau(\\pi, \\pi^*))`.

    Parameters
    ----------
    rank_utils : RankingUtils
    theta : float
        Concentration (larger = closer to the center).
    sampler : {"rim", "legacy"}
        ``rim`` (default) samples exactly with the repeated insertion model. ``legacy`` reproduces the
        paper's sampler (draw a distance from the exact distance distribution, then a permutation at that
        distance: uniformly by enumeration for ``d < 10``, by random adjacent swaps otherwise).
    rng : numpy Generator, optional
    """

    def __init__(self, rank_utils: RankingUtils, theta: float, sampler: str = "rim",
                 rng: Optional[np.random.Generator] = None):
        if sampler not in ("rim", "legacy"):
            raise ValueError(f"Unknown sampler {sampler!r}")
        self.rank_utils = rank_utils
        self.theta = float(theta)
        self.sampler = sampler
        self.rng = np.random.default_rng() if rng is None else rng
        d = len(rank_utils.items)
        dist_counts = rank_utils.get_dist_counts()
        log_Z = self.log_partition_fun(self.theta, d)
        k = np.arange(len(dist_counts))
        with np.errstate(divide="ignore"):
            log_p = np.log(dist_counts) - self.theta * k - log_Z
        dist_probs = np.exp(log_p)
        self.dist_counts = dist_counts
        self.dist_probs = dist_probs / dist_probs.sum()
        self._perm_dict_center = None
        self.dict_dist_perm = None

    # ------------------------------------------------------------------ sampling
    def sample_rim(self, center_pi: Ranking) -> list[int]:
        """Exact sample via the repeated insertion model, O(d^2)."""
        center = list(center_pi.permutation)
        d = len(center)
        out: list[int] = []
        for i, item in enumerate(center):
            # inserting at position j (0..i) creates i - j inversions
            j = np.arange(i + 1)
            w = np.exp(-self.theta * (i - j))
            pos = self.rng.choice(i + 1, p=w / w.sum())
            out.insert(int(pos), item)
        assert len(out) == d
        return out

    def sample(self, center_pi: Ranking):
        """Draw one ranking. Returns ``(ranking, err, distance)`` (``err`` is kept for compatibility)."""
        if self.sampler == "rim":
            s = Ranking(self.sample_rim(center_pi), self.rank_utils)
            f = int(round(self.rank_utils.kendall_tau_distance(s, center_pi)))
            return s, 0.0, f
        return self._sample_legacy(center_pi)

    def _sample_legacy(self, center_pi: Ranking):
        p = self.rank_utils.num_pairs
        f = int(np.nonzero(self.rng.multinomial(1, self.dist_probs))[0][0])
        if len(center_pi) < 10:
            if self._perm_dict_center != center_pi.permutation:
                self.build_perm_dict(center_pi)
            perms = self.dict_dist_perm[f]
            best_s = perms[self.rng.integers(len(perms))]
        else:
            center = center_pi
            if f > p / 2:
                center = center_pi.reverse()
                f_ = p - f
            else:
                f_ = f
            best_s = self.get_random_perm_at_dist_k(center, f_)
        best_s = Ranking(best_s, self.rank_utils)
        err = abs(f - self.rank_utils.kendall_tau_distance(best_s, center_pi))
        return best_s, float(err), f

    def build_perm_dict(self, center_pi: Ranking):
        dict_dist_perm = defaultdict(list)
        for perm in itertools.permutations(center_pi.permutation):
            perm_r = Ranking(perm, self.rank_utils)
            dist = int(round(self.rank_utils.kendall_tau_distance(perm_r, center_pi)))
            dict_dist_perm[dist].append(list(perm_r))
        self.dict_dist_perm = dict_dist_perm
        self._perm_dict_center = list(center_pi.permutation)

    def get_random_perm_at_dist_k(self, center_pi: Ranking, k: int) -> list[int]:
        """Legacy heuristic: ``k`` swaps of adjacent, never-swapped pairs (each adds one inversion)."""
        d = len(center_pi)
        s = list(center_pi.permutation)
        pairs_taken = np.zeros((d, d), dtype=bool)
        history = defaultdict(int)
        pairs = [(s[i], s[i + 1]) for i in range(d - 1)]
        for _ in range(k):
            if not pairs:
                break
            probs = np.exp(-np.array([history[(x, y)] for x, y in pairs], dtype=float))
            x, y = pairs[int(np.argmax(self.rng.multinomial(1, probs / probs.sum())))]
            for a, b in pairs:
                history[(a, b)] += 1
                history[(b, a)] += 1
            i, j = s.index(x), s.index(y)
            s[i], s[j] = s[j], s[i]
            pairs_taken[x, y] = pairs_taken[y, x] = True
            pairs = [(s[i], s[i + 1]) for i in range(d - 1) if not pairs_taken[s[i], s[i + 1]]]
        return s

    def estimate_accuracy(self, n_samples: int, center_pi: Ranking) -> float:
        """Probability that a sample keeps the center's top item first (Monte Carlo)."""
        y = center_pi[0]
        return float(np.mean([self.sample(center_pi)[0].permutation[0] == y for _ in range(n_samples)]))

    # ------------------------------------------------------------------ partition function and moments
    @staticmethod
    def log_partition_fun(theta: float, d: int) -> float:
        out = 0.0
        for k in range(1, d):
            out += log((1 - exp(-(k + 1) * theta)) / (1 - exp(-theta)))
        return out

    @staticmethod
    def d_log_partition(theta: float, d: int) -> float:
        out = 0.0
        for k in range(1, d):
            out += (k + 1) / (exp((k + 1) * theta) - 1) - 1 / (exp(theta) - 1)
        return out

    @staticmethod
    def d_d_log_partition(theta: float, d: int) -> float:
        out = 0.0
        for k in range(1, d):
            a = exp((k + 1) * theta)
            b = exp(theta)
            out += -((k + 1) ** 2 * a) / ((a - 1) ** 2) + b / ((b - 1) ** 2)
        return out

    @staticmethod
    def expected_distance(theta: float, d: int) -> float:
        """:math:`\\mathbb{E}_\\theta[d_\\tau(\\pi, \\pi^*)] = -\\partial_\\theta \\log Z(\\theta)`."""
        return -Mallows.d_log_partition(theta, d)

    @staticmethod
    def estimate_theta(d: int, mu_hat: float, bounds=(0.1, 10.0)) -> float:
        """Invert the expected distance: find theta with ``E_theta[d] = mu_hat`` (clipped to ``bounds``)."""
        lo, hi = bounds
        f = lambda t: Mallows.expected_distance(t, d) - mu_hat  # noqa: E731 (decreasing in t)
        if f(lo) <= 0:
            return lo
        if f(hi) >= 0:
            return hi
        return float(scipy.optimize.brentq(f, lo, hi, xtol=1e-12))

    @staticmethod
    def find_eq_theta_to_p0(d: int, p_0: float):
        """Theta for which the probability of keeping the top item equals ``p_0``."""
        D = RankingUtils(d).build_dist_counts()
        C = [D[d - 1][i] for i in range(((d - 1) * (d - 2)) // 2 + 1)]

        def A1(theta):
            return sum(C[i] * exp(-i * theta) for i in range(len(C)))

        def fun(theta):
            return 0.5 * (log(A1(theta)) - Mallows.log_partition_fun(theta, d) - log(p_0)) ** 2

        out = scipy.optimize.minimize_scalar(fun, bounds=(0.01, 10), method="bounded", options={"xatol": 1e-10})
        theta = float(out["x"])
        return theta, A1(theta) / exp(Mallows.log_partition_fun(theta, d))

