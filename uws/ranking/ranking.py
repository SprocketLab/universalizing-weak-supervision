"""Ranking objects, pair-sign encodings and Kendall tau distances (full and partial rankings)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Iterable, Optional

import numpy as np

# Penalties used by the original (paper) implementation of the partial Kendall tau distance.
LEGACY_PARTIAL_PENALTY = 0.1


class RankingUtils:
    """Helpers for rankings over ``d`` items ``0..d-1``.

    A ranking is encoded by its *pair signs* ``z``: one entry per unordered pair ``(i, j), i < j``
    (ordered as in :attr:`unique_pairs`), ``+1`` if ``i`` is ranked before ``j``, ``-1`` otherwise and
    ``0`` if the pair is unobserved (masked).

    Parameters
    ----------
    d : int
        Number of items.
    p : float
        Penalty of Fagin et al.'s :math:`K^{(p)}` distance for a pair ordered by one ranking only.
    legacy : bool
        Reproduce the distance used in the paper's experiments (penalty 0.1 for pairs observed by one
        ranking and ``1/#pairs`` for pairs observed by neither). Full rankings are unaffected.
    """

    def __init__(self, d: int, p: float = 0.5, legacy: bool = False):
        self.d = d
        self.p = p
        self.legacy = legacy
        self.items = list(range(d))
        self.unique_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        self.pair_i = np.array([i for i, _ in self.unique_pairs], dtype=int)
        self.pair_j = np.array([j for _, j in self.unique_pairs], dtype=int)
        self.pair_index_map = {self.pair_key(i, j): k for k, (i, j) in enumerate(self.unique_pairs)}
        self.dist_counts = None

    def __repr__(self) -> str:
        return f"RankingUtils(d={self.d}, p={self.p}, legacy={self.legacy})"

    @property
    def num_pairs(self) -> int:
        return len(self.unique_pairs)

    def get_unique_pairs(self):
        return self.unique_pairs

    @staticmethod
    def pair_key(a, b) -> str:
        return f"{a},{b}"

    def get_random_ranking(self, rng: Optional[np.random.Generator] = None) -> Ranking:
        rng = np.random.default_rng() if rng is None else rng
        return Ranking(list(rng.permutation(self.d)), self)

    # ------------------------------------------------------------------ encodings
    def ranking_to_pair_signs(self, r) -> np.ndarray:
        """Pair signs of a :class:`Ranking` (respecting its mask) or of a plain permutation."""
        perm = r.permutation if isinstance(r, Ranking) else r
        pos = np.empty(self.d, dtype=int)
        pos[np.asarray(perm, dtype=int)] = np.arange(self.d)
        z = np.where(pos[self.pair_i] < pos[self.pair_j], 1.0, -1.0)
        if isinstance(r, Ranking):
            z = z * r.mask
        return z

    def Z(self, r1, r2) -> np.ndarray:
        z1 = r1.z if isinstance(r1, Ranking) else self.ranking_to_pair_signs(r1)
        z2 = r2.z if isinstance(r2, Ranking) else self.ranking_to_pair_signs(r2)
        return np.multiply(z1, z2)

    # ------------------------------------------------------------------ distances
    def pair_distances(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Per-pair Kendall contributions for (broadcastable) pair-sign arrays; last axis = pairs."""
        z1, z2 = np.asarray(z1, dtype=float), np.asarray(z2, dtype=float)
        o1, o2 = z1 != 0, z2 != 0
        both = o1 & o2
        one = o1 ^ o2
        D = np.where(both & (z1 * z2 < 0), 1.0, 0.0)
        if self.legacy:
            D = D + np.where(one, LEGACY_PARTIAL_PENALTY, 0.0) + np.where(~o1 & ~o2, 1.0 / z1.shape[-1], 0.0)
        else:
            D = D + np.where(one, self.p, 0.0)
        return D

    def z_distance(self, z1: np.ndarray, z2: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Kendall tau (:math:`K^{(p)}`) distance between pair-sign arrays (broadcast over leading axes)."""
        D = self.pair_distances(z1, z2).sum(-1)
        if normalize:
            n = ((np.asarray(z1) != 0) | (np.asarray(z2) != 0)).sum(-1)
            D = np.where(n > 0, D / np.maximum(n, 1), 0.0)
        return D

    def kendall_tau_distance(self, r1: Ranking, r2: Ranking, normalize: bool = False) -> float:
        """Kendall tau distance between two rankings.

        For full rankings this is the number of discordant pairs. For partial rankings it is Fagin et
        al.'s :math:`K^{(p)}`: discordant pairs observed by both rankings count 1, pairs observed by only
        one count ``p`` and pairs observed by neither count 0. ``normalize`` divides by the number of
        pairs observed by at least one ranking (0 if there are none).
        """
        return float(self.z_distance(r1.z, r2.z, normalize))

    def get_pair_wise_dists(self, lstRanks: Sequence[Ranking]) -> np.ndarray:
        """Matrix of pairwise Kendall tau distances between a list of rankings."""
        Z = np.stack([r.z for r in lstRanks])
        D = self.z_distance(Z[:, None, :], Z[None, :, :])
        np.fill_diagonal(D, 0.0)  # (legacy mode charges unobserved pairs even against itself)
        return D

    def mean_kt_distance(self, Y1, Y2, normalize: bool = True) -> float:
        """Mean Kendall tau distance between two aligned lists of rankings."""
        Z1 = np.stack([y.z for y in Y1])
        Z2 = np.stack([y.z for y in Y2])
        return float(np.mean(self.z_distance(Z1, Z2, normalize)))

    # ------------------------------------------------------------------ aggregation (see aggregation.py)
    def kemeny(self, lstRanks, D=None, weights=None, exact_max_d: int = 12):
        """(Weighted) Kemeny ranking: the permutation minimizing the weighted sum of Kendall distances to
        ``lstRanks``. ``D`` is ignored unless ``legacy=True`` (where the medoid is returned instead)."""
        from .aggregation import kemeny

        if self.legacy:
            return self.kemeny_medoid(lstRanks, D, weights)
        return kemeny(lstRanks, self, weights=weights, exact_max_d=exact_max_d)

    def weighted_kemeny(self, lstRanks, weights, D=None, exact_max_d: int = 12):
        return self.kemeny(lstRanks, D=D, weights=weights, exact_max_d=exact_max_d)

    def kemeny_medoid(self, lstRanks, D=None, weights=None):
        """The input ranking minimizing the (weighted) mean distance to the others (paper's 'kemeny')."""
        from .aggregation import kemeny_medoid

        return kemeny_medoid(lstRanks, self, D=D, weights=weights)

    def weighted_kemeny_medoid(self, lstRanks, weights, D=None):
        return self.kemeny_medoid(lstRanks, D=D, weights=weights)

    def pair_wise_majority(self, lstRanks, weights=None):
        from .aggregation import pairwise_majority

        return pairwise_majority(lstRanks, self, weights=weights)

    def pos_est_majority(self, lstRanks, weights=None, rng=None):
        from .aggregation import position_estimation

        return position_estimation(lstRanks, self, weights=weights, rng=rng)

    def z_to_ranking(self, z, w=None, confidence=None, rng=None):
        from .aggregation import z_to_ranking

        return z_to_ranking(z, self, w=w, confidence=confidence, rng=rng)

    # ------------------------------------------------------------------ Mallows helpers
    def build_dist_counts(self) -> np.ndarray:
        """``D[k][j]``: number of permutations of ``k`` items at Kendall distance ``j`` from a fixed one."""
        d = len(self.items)
        p = (d * (d - 1)) // 2
        D = np.zeros((max(d, 2) + 2, p + 1))
        D[1][0] = 1
        D[2][0] = 1
        D[2][1] = 1
        for i in range(3, d + 1):
            prev = 0
            k = (i * (i - 1)) // 2
            for j in range(k + 1):
                if j < i:
                    D[i][j] = prev + D[i - 1][j]
                else:
                    D[i][j] = prev + D[i - 1][j] - D[i - 1][j - i]
                prev = D[i][j]
        return D

    def get_dist_counts(self) -> np.ndarray:
        if self.dist_counts is None:
            self.dist_counts = self.build_dist_counts()
        return self.dist_counts[len(self.items), :]

    # ------------------------------------------------------------------ permutation <-> integer
    def set_perm2int_int2perm_mapping(self):
        perms = list(itertools.permutations(range(self.d)))
        self.perm2int_map = {tuple(p): i for i, p in enumerate(perms)}
        self.int2perm_map = {i: list(p) for i, p in enumerate(perms)}
        return self.perm2int_map, self.int2perm_map

    def perm2int(self, lstRanks) -> list[int]:
        return [self.perm2int_map[tuple(int(x) for x in r.permutation)] for r in lstRanks]

    def int2perm(self, lstInts) -> list[Ranking]:
        return [Ranking(self.int2perm_map[int(i)], r_utils=self) for i in lstInts]


class Ranking(Sequence):
    """A ranking (permutation of items ``0..d-1``, best first) with an optional mask of unobserved pairs.

    Parameters
    ----------
    permutation : sequence of int
        Items in ranked order.
    r_utils : RankingUtils, optional
        Shared helper; created on the fly if omitted.
    z : array, optional
        Precomputed pair signs.
    """

    def __init__(self, permutation: Iterable[int], r_utils: Optional[RankingUtils] = None, z=None):
        self.permutation = [int(x) for x in permutation]
        self.r_utils = RankingUtils(len(self.permutation)) if r_utils is None else r_utils
        self.mask = np.ones(self.r_utils.num_pairs)
        self.mask_permutation = [0] * len(self.permutation)
        self.z = self.r_utils.ranking_to_pair_signs(self) if z is None else np.asarray(z, dtype=float)

    def __len__(self) -> int:
        return len(self.permutation)

    def __getitem__(self, i):
        return self.permutation[i]

    def __str__(self) -> str:
        return str(self.permutation)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Ranking):
            return NotImplemented
        return self.permutation == other.permutation and np.array_equal(self.z, other.z)

    __hash__ = None  # mutable

    def copy(self) -> Ranking:
        r = Ranking(self.permutation, self.r_utils, z=self.z.copy())
        r.mask = self.mask.copy()
        r.mask_permutation = list(self.mask_permutation)
        return r

    @property
    def is_partial(self) -> bool:
        return bool(np.any(self.z == 0))

    def reverse(self) -> Ranking:
        return Ranking(list(reversed(self.permutation)), self.r_utils)

    def mask_items(self, lst_items, rng: Optional[np.random.Generator] = None) -> None:
        """Mask ``lst_items``: every pair involving one of them becomes unobserved (sign 0).

        The masked items are moved, in random order, to the end of :attr:`permutation`, and
        :attr:`mask_permutation` flags them with 1.
        """
        lst_items = [int(x) for x in lst_items]
        masked = set(lst_items)
        for k, (i, j) in enumerate(self.r_utils.unique_pairs):
            if i in masked or j in masked:
                self.mask[k] = 0
        kept = [i for i in self.permutation if i not in masked]
        self.z = self.r_utils.ranking_to_pair_signs(self)
        rng = np.random.default_rng() if rng is None else rng
        tail = [lst_items[i] for i in rng.permutation(len(lst_items))]
        self.permutation = kept + tail
        self.mask_permutation = [0] * len(kept) + [1] * len(tail)


def perm2ranking(Y, d: int) -> list[list[int]]:
    """Invert permutations: ``perm[k] = item`` (item at rank k) -> ``rank[item] = k``."""
    out = []
    for y in Y:
        perm = y.permutation if isinstance(y, Ranking) else y
        rank = [0] * d
        for k, item in enumerate(perm):
            rank[item] = k
        out.append(rank)
    return out


def ranking_to_score(Y, d: int, highest_first: bool):
    """Rankings -> per-item scores (torch tensor) for learning-to-rank."""
    import torch

    Y_rank = torch.tensor(perm2ranking(Y, d), dtype=torch.float32)
    return d - Y_rank if highest_first else Y_rank
