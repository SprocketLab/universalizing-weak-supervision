"""Rank aggregation rules used for label-model inference."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Sequence

import numpy as np

from .digraph import RankingDiGraph
from .ranking import Ranking, RankingUtils


def _weights(lstRanks, weights) -> np.ndarray:
    return np.ones(len(lstRanks)) if weights is None else np.asarray(weights, dtype=float)[: len(lstRanks)]


def preference_matrix(lstRanks: Sequence[Ranking], r_utils: RankingUtils, weights=None) -> np.ndarray:
    """``W[a, b]`` = total weight of rankings placing ``a`` before ``b`` (unobserved pairs add nothing)."""
    w = _weights(lstRanks, weights)
    Z = np.stack([r.z for r in lstRanks])  # (m, P)
    plus = (w[:, None] * (Z > 0)).sum(0)
    minus = (w[:, None] * (Z < 0)).sum(0)
    W = np.zeros((r_utils.d, r_utils.d))
    W[r_utils.pair_i, r_utils.pair_j] = plus
    W[r_utils.pair_j, r_utils.pair_i] = minus
    return W


def kemeny_cost(perm: Sequence[int], W: np.ndarray) -> float:
    """Weighted disagreement of ``perm`` with preference matrix ``W``."""
    perm = np.asarray(perm)
    pos = np.empty(len(perm), dtype=int)
    pos[perm] = np.arange(len(perm))
    after = pos[:, None] < pos[None, :]  # a placed before b
    return float((W.T * after).sum())  # pay W[b, a] whenever a is before b


def _kemeny_exact(W: np.ndarray) -> list[int]:
    """Exact Kemeny optimum by dynamic programming over subsets, O(2^d d)."""
    d = W.shape[0]
    full = (1 << d) - 1
    colsum = W.sum(0)  # colsum[v] = sum_u W[u, v]
    # in_s[S, v] = sum_{u in S} W[u, v]
    in_s = np.zeros((1 << d, d))
    for S in range(1, 1 << d):
        low = S & -S
        u = low.bit_length() - 1
        in_s[S] = in_s[S ^ low] + W[u]
    best = np.full(1 << d, np.inf)
    best[0] = 0.0
    choice = np.full(1 << d, -1, dtype=int)
    for S in range(1 << d):
        if not np.isfinite(best[S]):
            continue
        rem = full ^ S
        v_bits = rem
        while v_bits:
            low = v_bits & -v_bits
            v = low.bit_length() - 1
            v_bits ^= low
            # v goes next; every u still remaining after v pays W[u, v]
            cost = best[S] + colsum[v] - in_s[S, v] - W[v, v]
            T = S | low
            if cost < best[T] - 1e-12:
                best[T] = cost
                choice[T] = v
    perm = []
    S = full
    while S:
        v = choice[S]
        perm.append(v)
        S ^= 1 << v
    return perm[::-1]


def _kemeny_local_search(W: np.ndarray, max_iter: int = 1000) -> list[int]:
    """Borda initialization followed by insertion local search (a strong Kemeny heuristic)."""
    d = W.shape[0]
    perm = list(np.argsort(-(W.sum(1) - W.sum(0)), kind="stable"))
    cost = kemeny_cost(perm, W)
    for _ in range(max_iter):
        improved = False
        for i in range(d):
            item = perm[i]
            rest = perm[:i] + perm[i + 1 :]
            for j in range(d):
                if j == i:
                    continue
                cand = rest[:j] + [item] + rest[j:]
                c = kemeny_cost(cand, W)
                if c < cost - 1e-12:
                    perm, cost, improved = cand, c, True
                    break
            if improved:
                break
        if not improved:
            break
    return [int(x) for x in perm]


def kemeny(lstRanks: Sequence[Ranking], r_utils: RankingUtils, weights=None, exact_max_d: int = 12) -> Ranking:
    """(Weighted) Kemeny aggregation: exact for ``d <= exact_max_d``, local search otherwise."""
    W = preference_matrix(lstRanks, r_utils, weights)
    perm = _kemeny_exact(W) if r_utils.d <= exact_max_d else _kemeny_local_search(W)
    return Ranking(perm, r_utils)


def kemeny_medoid(lstRanks: Sequence[Ranking], r_utils: RankingUtils, D=None, weights=None) -> Ranking:
    """Return (a copy of) the input ranking with the smallest weighted mean distance to the others."""
    if D is None:
        D = r_utils.get_pair_wise_dists(lstRanks)
    w = _weights(lstRanks, weights)
    mu = (np.asarray(D)[: len(lstRanks), : len(lstRanks)] * w[None, :]).mean(axis=1)
    return lstRanks[int(np.argmin(mu))].copy()


def _signs(z_agg: np.ndarray) -> np.ndarray:
    return np.sign(z_agg)


def pairwise_majority(lstRanks: Sequence[Ranking], r_utils: RankingUtils, weights=None, rng=None) -> Ranking:
    """(Weighted) pairwise majority vote, turned into a ranking through a greedy topological sort."""
    w = _weights(lstRanks, weights)
    Z = np.stack([r.z for r in lstRanks])
    z_agg = (w[:, None] * Z).sum(0)
    confidence = (np.abs(w[:, None] * Z)).sum(0)
    return z_to_ranking(_signs(z_agg), r_utils, z_agg, confidence, rng=rng)


def z_to_ranking(z, r_utils: RankingUtils, w=None, confidence=None, rng=None) -> Ranking:
    """Build a digraph from pair signs ``z`` and topologically sort it into a ranking."""
    rdag = RankingDiGraph(rng=rng)
    rdag.nodes = copy.deepcopy(r_utils.items)
    w = np.zeros(len(z)) if w is None else w
    confidence = np.zeros(len(z)) if confidence is None else confidence
    for i in range(len(z)):
        u, v = r_utils.unique_pairs[i]
        if z[i] == 1:
            rdag.add_edge(u, v, w[i], confidence[i])
        elif z[i] == -1:
            rdag.add_edge(v, u, abs(w[i]), confidence[i])
        elif z[i] == 0 and confidence[i] > 0:
            rdag.add_edge(u, v, w[i], confidence[i])
            rdag.add_edge(v, u, w[i], confidence[i])
    return Ranking(rdag.topo_sort(), r_utils)


def position_estimation(lstRanks: Sequence[Ranking], r_utils: RankingUtils, weights=None, rng=None) -> Ranking:
    """Rank items by how many pairwise (weighted) majority contests they lose or tie."""
    rng = np.random.default_rng() if rng is None else rng
    w = _weights(lstRanks, weights)
    Z = np.stack([r.z for r in lstRanks])
    z = _signs((w[:, None] * Z).sum(0))
    positions = defaultdict(list)
    for i in r_utils.items:
        losses = 0
        for j in r_utils.items:
            if i < j:
                losses += z[r_utils.pair_index_map[r_utils.pair_key(i, j)]] <= 0
            elif i > j:
                losses += z[r_utils.pair_index_map[r_utils.pair_key(j, i)]] >= 0
        positions[int(losses)].append(i)
    pi_hat = []
    for k in sorted(positions):
        group = positions[k]
        pi_hat.extend(group[i] for i in rng.permutation(len(group)))
    return Ranking(pi_hat, r_utils)
