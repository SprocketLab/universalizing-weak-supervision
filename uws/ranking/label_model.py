"""Label model for (full or partial) rankings: Mallows LFs, triplet parameter estimation."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np

from ..base import BaseLabelModel
from ..config import InferenceRule, RankingConfig, TrainMethod
from ..utils.triplets import individual_distances, rank_one_fit
from . import aggregation
from .mallows import Mallows
from .ranking import Ranking, RankingUtils

logger = logging.getLogger(__name__)


def as_rankings(L, r_utils: Optional[RankingUtils] = None) -> list[list[Ranking]]:
    """Accept ``L`` as nested lists of :class:`Ranking` or an int array of permutations ``(n, m, d)``."""
    if isinstance(L, np.ndarray):
        if L.ndim != 3:
            raise ValueError(f"Expected an (n, m, d) permutation array, got shape {L.shape}")
        r_utils = RankingUtils(L.shape[2]) if r_utils is None else r_utils
        return [[Ranking(p, r_utils) for p in row] for row in L]
    return L


def sign_expected_distances(Z: np.ndarray, groups: Optional[Sequence] = None) -> np.ndarray:
    """Per-LF expected Kendall distance to the latent ranking from pair-sign agreements.

    With ``z`` the pair signs, conditional independence gives ``M_ab = E[z_a z_b] = s_a s_b``, where
    ``s_a = E[z_a z_y]`` is LF ``a``'s sign agreement with the latent ranking. ``s`` is fit as the rank-1
    factorization of the off-diagonal of ``M`` (least squares, weighted by the number of co-observed pairs,
    initialized from the leading eigenvector), which uses every pair and, unlike per-triplet ratios, is not
    dominated by noise from near-random partners. Agreements are averaged over the pairs both LFs observe, so
    masking does not bias the estimate; the returned value is the expected distance of the full (unmasked)
    LF ranking, ``#pairs * (1 - s_a) / 2``.

    Parameters
    ----------
    Z : (n, m, P) array
        Pair signs of every LF on every example (0 = unobserved).
    groups : sequence, optional
        LF families; pairs within a family are excluded from the fit (their errors are correlated).
    """
    n, m, P = Z.shape
    num = np.einsum("nap,nbp->ab", Z, Z)
    cnt = np.einsum("nap,nbp->ab", np.abs(Z), np.abs(Z))
    M = np.divide(num, cnt, out=np.zeros_like(num), where=cnt > 0)
    W = cnt.astype(float)
    if groups is not None:
        g = np.asarray(groups)
        W[g[:, None] == g[None, :]] = 0.0
    if np.any((W - np.diag(np.diag(W))).sum(1) == 0):
        raise ValueError("Some LF shares no observed pairs with any usable partner")
    s = rank_one_fit(M, W)
    return P * (1.0 - s) / 2.0


class RankingLabelModel(BaseLabelModel):
    """Weak supervision for rankings (Shin et al., ICLR 2022).

    Each LF is modeled as a Mallows distribution centered at the latent ranking. The expected distance
    of each LF to the latent ranking is estimated from LF-LF Kendall distances with triplets, mapped to a
    Mallows ``theta`` and used as aggregation weight.

    Parameters
    ----------
    r_utils : RankingUtils
        Shared ranking helper (fixes ``d``, the partial-ranking penalty and legacy mode).
    train_method : str or TrainMethod
        ``triplet``, ``median_triplet``, ``lstsq``, ``sign_triplet`` and their ``*_opt`` variants (exact Mallows
        inversion); see :class:`uws.config.TrainMethod`.
    inference_rule : str or InferenceRule
        Aggregation rule used by :meth:`predict`.
    theta_bounds : (float, float)
        Clip range for estimated thetas.
    seed : int, optional
        Seed of the random tie-breaking in aggregation.
    """

    def __init__(self, r_utils: RankingUtils, train_method="median_triplet_opt",
                 inference_rule="weighted_kemeny", theta_bounds=(0.1, 100.0), seed: Optional[int] = None):
        self.r_utils = r_utils
        self.config = RankingConfig(train_method, inference_rule)
        self.theta_bounds = theta_bounds
        self.rng = np.random.default_rng(seed)
        self.thetas = None
        self.expected_dists = None
        self.D_lf = None

    def __repr__(self) -> str:
        return (f"RankingLabelModel(d={self.r_utils.d}, train_method={self.config.train_method.value!r}, "
                f"inference_rule={self.config.inference_rule.value!r})")

    # ------------------------------------------------------------------ fitting
    def lf_distance_matrix(self, L, numLFs: Optional[int] = None) -> np.ndarray:
        """Mean Kendall distance between every pair of LFs, ``(m, m)``."""
        m = len(L[0]) if numLFs is None else numLFs
        Z = np.stack([np.stack([L[i][j].z for j in range(m)]) for i in range(len(L))])  # (n, m, P)
        D = np.zeros((m, m))
        for a in range(m):
            D[a] = self.r_utils.z_distance(Z[:, a : a + 1, :], Z).mean(0)
        np.fill_diagonal(D, 0.0)
        return D

    def fit(self, L, numLFs: Optional[int] = None, groups: Optional[Sequence] = None) -> RankingLabelModel:
        """Estimate per-LF thetas from unlabeled LF rankings ``L`` (``n`` lists of ``m`` rankings)."""
        L = as_rankings(L, self.r_utils)
        m = len(L[0]) if numLFs is None else numLFs
        d = len(L[0][0])
        method = self.config.train_method
        self.D_lf = self.lf_distance_matrix(L, m)
        if method.estimator == "sign_triplet":
            Z = np.stack([np.stack([L[i][j].z for j in range(m)]) for i in range(len(L))])
            raw = sign_expected_distances(Z, groups)
        else:
            raw = individual_distances(self.D_lf, method.estimator, groups=groups, clip=False)
        if np.any(raw <= 0):
            warnings.warn(f"Non-positive expected distance estimates {np.round(raw, 4)} were clipped; "
                          "the triplet assumptions may be violated (e.g. correlated LFs).", stacklevel=2)
        self.expected_dists = np.maximum(raw, 1e-6)
        logger.info("expected_dists %s", self.expected_dists)
        if method.exact_theta:
            thetas = np.array([Mallows.estimate_theta(d, e) for e in self.expected_dists])
        else:
            thetas = 1.0 / self.expected_dists
        self.thetas = np.clip(thetas, *self.theta_bounds)
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, L, numLFs: Optional[int] = None, inference_rule=None, lst_D=None) -> list[Ranking]:
        """Aggregate the LF rankings of every example into one ranking."""
        L = as_rankings(L, self.r_utils)
        k = len(L[0]) if numLFs is None else numLFs
        rule = self.config.inference_rule if inference_rule is None else InferenceRule(inference_rule)
        if rule.weighted and self.thetas is None:
            raise RuntimeError("Call fit() before weighted inference")
        w = self.thetas[:k] if rule.weighted else None
        ru = self.r_utils
        legacy_kemeny = ru.legacy and rule in (InferenceRule.KEMENY, InferenceRule.WEIGHTED_KEMENY)
        out = []
        for i in range(len(L)):
            ranks = L[i][:k]
            if rule in (InferenceRule.KEMENY_MEDOID, InferenceRule.WEIGHTED_KEMENY_MEDOID) or legacy_kemeny:
                D = None if lst_D is None else lst_D[i][:k, :k]
                y = aggregation.kemeny_medoid(ranks, ru, D=D, weights=w)
            elif rule in (InferenceRule.KEMENY, InferenceRule.WEIGHTED_KEMENY):
                y = aggregation.kemeny(ranks, ru, weights=w)
            elif rule in (InferenceRule.PAIRWISE_MAJORITY, InferenceRule.WEIGHTED_PAIRWISE_MAJORITY):
                y = aggregation.pairwise_majority(ranks, ru, weights=w, rng=self.rng)
            else:
                y = aggregation.position_estimation(ranks, ru, weights=w, rng=self.rng)
            if not ru.legacy:
                y = Ranking(y.permutation, ru)  # the estimate is a full ranking
            out.append(y)
        return out

    def score(self, L, Y, normalize: bool = True) -> dict:
        Y_hat = self.predict(L)
        return {"mean_kt": self.r_utils.mean_kt_distance(Y_hat, Y, normalize)}

    # ------------------------------------------------------------------ backwards compatibility
    def train(self, conf: dict, L, numLFs: Optional[int] = None):
        """Deprecated: ``WeakSupRanking.train(conf, L)``; use ``fit``."""
        self.config = RankingConfig.from_dict({**{"inference_rule": self.config.inference_rule}, **conf})
        return self.fit(L, numLFs)

    def infer_ranking(self, conf: dict, L, numLFs: Optional[int] = None, lst_D=None):
        """Deprecated: ``WeakSupRanking.infer_ranking(conf, L)``; use ``predict``."""
        return self.predict(L, numLFs, inference_rule=conf.get("inference_rule"), lst_D=lst_D)


class WeakSupRanking(RankingLabelModel):
    """Backwards-compatible name: ``WeakSupRanking(r_utils).train(conf, L)``."""

    def __init__(self, r_utils: RankingUtils, **kw):
        super().__init__(r_utils, **kw)


__all__ = ["RankingLabelModel", "WeakSupRanking", "TrainMethod", "InferenceRule", "as_rankings",
           "sign_expected_distances"]
