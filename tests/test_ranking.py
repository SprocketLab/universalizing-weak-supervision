import itertools

import numpy as np
import pytest

from uws.ranking import Ranking, RankingDiGraph, RankingUtils, kemeny, kemeny_medoid, preference_matrix
from uws.ranking.aggregation import _kemeny_local_search, kemeny_cost


def brute_force_kt(p1, p2):
    pos1 = {x: i for i, x in enumerate(p1)}
    pos2 = {x: i for i, x in enumerate(p2)}
    return sum((pos1[a] - pos1[b]) * (pos2[a] - pos2[b]) < 0 for a, b in itertools.combinations(p1, 2))


def test_pair_signs_convention():
    ru = RankingUtils(3)
    r = Ranking([2, 0, 1], ru)
    # pairs (0,1), (0,2), (1,2): 0 before 1 -> +1; 2 before 0 -> -1; 2 before 1 -> -1
    assert list(r.z) == [1, -1, -1]


def test_full_kendall_matches_brute_force(rng):
    ru = RankingUtils(7)
    for _ in range(50):
        a, b = rng.permutation(7), rng.permutation(7)
        assert ru.kendall_tau_distance(Ranking(a, ru), Ranking(b, ru)) == brute_force_kt(list(a), list(b))


def test_partial_kendall_kp():
    ru = RankingUtils(4, p=0.5)
    r1, r2 = Ranking([0, 1, 2, 3], ru), Ranking([1, 0, 2, 3], ru)
    r2.mask_items([3])
    # pairs with item 3 are observed by r1 only -> 3 * p; (0,1) discordant -> 1
    assert ru.kendall_tau_distance(r1, r2) == pytest.approx(1 + 3 * 0.5)
    # normalization by pairs observed by at least one ranking (all 6 here)
    assert ru.kendall_tau_distance(r1, r2, normalize=True) == pytest.approx(2.5 / 6)
    # pairs unobserved by both contribute nothing
    r1.mask_items([3])
    assert ru.kendall_tau_distance(r1, r2) == pytest.approx(1.0)


def test_legacy_kendall():
    ru = RankingUtils(4, legacy=True)
    r1, r2 = Ranking([0, 1, 2, 3], ru), Ranking([1, 0, 2, 3], ru)
    r1.mask_items([3])
    r2.mask_items([3])
    # 3 pairs unobserved by both: 1/6 each; one discordant pair
    assert ru.kendall_tau_distance(r1, r2) == pytest.approx(1 + 3 / 6)


def test_normalize_with_no_observed_pairs():
    ru = RankingUtils(3)
    r1, r2 = Ranking([0, 1, 2], ru), Ranking([0, 1, 2], ru)
    r1.mask_items([0, 1, 2])
    r2.mask_items([0, 1, 2])
    assert ru.kendall_tau_distance(r1, r2, normalize=True) == 0.0


def test_mask_items_does_not_mutate_input(rng):
    ru = RankingUtils(5)
    r = Ranking([4, 3, 2, 1, 0], ru)
    items = [0, 3]
    r.mask_items(items, rng)
    assert items == [0, 3]
    assert r.permutation[:3] == [4, 2, 1]
    assert r.mask_permutation == [0, 0, 0, 1, 1]
    assert sorted(r.permutation) == list(range(5))


def test_pairwise_distance_matrix(rng):
    ru = RankingUtils(5)
    rs = [ru.get_random_ranking(rng) for _ in range(4)]
    D = ru.get_pair_wise_dists(rs)
    for i, j in itertools.product(range(4), repeat=2):
        assert D[i, j] == ru.kendall_tau_distance(rs[i], rs[j])


@pytest.mark.parametrize("d", [3, 5, 7])
def test_exact_kemeny_matches_brute_force(rng, d):
    ru = RankingUtils(d)
    for _ in range(5):
        rs = [ru.get_random_ranking(rng) for _ in range(5)]
        w = rng.uniform(0.1, 3, size=5)
        W = preference_matrix(rs, ru, w)
        best = min(kemeny_cost(p, W) for p in itertools.permutations(range(d)))
        assert kemeny_cost(kemeny(rs, ru, w).permutation, W) == pytest.approx(best)
        assert kemeny_cost(_kemeny_local_search(W), W) >= best - 1e-9


def test_kemeny_cost_equals_weighted_distance_sum(rng):
    ru = RankingUtils(6)
    rs = [ru.get_random_ranking(rng) for _ in range(4)]
    w = np.array([1.0, 2.0, 0.5, 1.5])
    y = ru.get_random_ranking(rng)
    expected = sum(wi * ru.kendall_tau_distance(y, r) for wi, r in zip(w, rs))
    assert kemeny_cost(y.permutation, preference_matrix(rs, ru, w)) == pytest.approx(expected)


def test_kemeny_medoid_returns_copy(rng):
    ru = RankingUtils(4)
    rs = [ru.get_random_ranking(rng) for _ in range(3)]
    med = kemeny_medoid(rs, ru)
    assert any(med == r for r in rs)
    assert all(med is not r for r in rs)


def test_digraph_from_ranking_and_mask_node():
    g = RankingDiGraph([2, 0, 1])
    assert len(g.out_edges[2]) == 2 and len(g.in_edges[1]) == 2
    g.mask_node(0)
    assert all(e.u != 0 and e.v != 0 for es in g.out_edges.values() for e in es)
    assert g.get_nodes_with_no_in_edges() == [2, 0]


def test_perm_int_roundtrip():
    ru = RankingUtils(4)
    ru.set_perm2int_int2perm_mapping()
    rs = [Ranking([3, 1, 0, 2], ru), Ranking([0, 1, 2, 3], ru)]
    assert [r.permutation for r in ru.int2perm(ru.perm2int(rs))] == [r.permutation for r in rs]
