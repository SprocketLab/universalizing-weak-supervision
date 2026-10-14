import itertools

import numpy as np
import pytest

from uws.ranking import Mallows, Ranking, RankingUtils


def test_dist_counts_are_mahonian():
    ru = RankingUtils(5)
    counts = ru.get_dist_counts()
    brute = np.zeros(11)
    center = Ranking(range(5), ru)
    for p in itertools.permutations(range(5)):
        brute[int(ru.kendall_tau_distance(Ranking(p, ru), center))] += 1
    assert np.array_equal(counts, brute)


@pytest.mark.parametrize("theta", [0.3, 1.0, 2.5])
def test_rim_matches_exact_distance_distribution(theta):
    ru = RankingUtils(6)
    m = Mallows(ru, theta, rng=np.random.default_rng(1))
    center = Ranking([3, 1, 5, 0, 2, 4], ru)
    ds = np.array([m.sample(center)[2] for _ in range(20000)])
    emp = np.bincount(ds, minlength=len(m.dist_probs)) / len(ds)
    assert np.abs(emp - m.dist_probs).max() < 0.012


def test_rim_large_d_mean_distance():
    d, theta = 15, 0.5
    ru = RankingUtils(d)
    m = Mallows(ru, theta, rng=np.random.default_rng(2))
    center = ru.get_random_ranking(np.random.default_rng(3))
    ds = [m.sample(center)[2] for _ in range(3000)]
    assert np.mean(ds) == pytest.approx(Mallows.expected_distance(theta, d), rel=0.03)


def test_legacy_sampler_still_works():
    ru = RankingUtils(5)
    m = Mallows(ru, 1.0, sampler="legacy", rng=np.random.default_rng(0))
    center = Ranking(range(5), ru)
    s, err, f = m.sample(center)
    assert err == 0 and ru.kendall_tau_distance(s, center) == f


@pytest.mark.parametrize("theta", [0.2, 0.8, 3.0])
def test_estimate_theta_roundtrip(theta):
    d = 8
    assert Mallows.estimate_theta(d, Mallows.expected_distance(theta, d)) == pytest.approx(theta, rel=1e-6)


def test_find_eq_theta_to_p0():
    theta, p0 = Mallows.find_eq_theta_to_p0(6, 0.5)
    assert p0 == pytest.approx(0.5, abs=1e-4) and theta > 0
