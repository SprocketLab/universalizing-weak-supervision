import numpy as np
import pytest

from uws.ranking import RankingLabelModel, RankingUtils, WeakSupRanking, generate_true_rankings, sample_mallows_LFs
from uws.utils import individual_distances


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    ru = RankingUtils(6)
    Y = generate_true_rankings(6, 400, rng, ru)
    thetas = np.array([0.3, 1.5, 0.6, 2.0, 0.4, 1.0])
    return ru, Y, thetas, sample_mallows_LFs(Y, len(thetas), thetas, rng=rng)


def test_individual_distances_exact_on_additive():
    e = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    E = e[:, None] + e[None, :]
    np.fill_diagonal(E, 0)
    for method in ("triplet", "median_triplet", "mean_triplet", "lstsq"):
        assert np.allclose(individual_distances(E, method), e)


def test_triplet_leftovers_are_estimated():
    e = np.array([1.0, 2.0, 0.5, 3.0])  # m % 3 != 0
    E = e[:, None] + e[None, :]
    assert np.allclose(individual_distances(E, "triplet"), e)


def test_triplets_need_three_lfs():
    with pytest.raises(ValueError):
        individual_distances(np.zeros((2, 2)))


@pytest.mark.parametrize("method", ["triplet_opt", "median_triplet_opt", "lstsq_opt"])
def test_theta_ordering_recovered(data, method):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru, train_method=method).fit(L)
    assert np.all(np.isfinite(lm.thetas))
    from scipy.stats import spearmanr

    assert spearmanr(lm.thetas, thetas).correlation > 0.9


@pytest.mark.parametrize("m", [4, 5])
def test_non_multiple_of_three(data, m):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru, train_method="triplet_opt").fit([row[:m] for row in L])
    assert len(lm.thetas) == m and np.all(np.isfinite(lm.thetas)) and np.all(lm.thetas > 0)


def test_weighted_beats_unweighted(data):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru).fit(L)
    weighted = ru.mean_kt_distance(lm.predict(L), Y)
    unweighted = ru.mean_kt_distance(lm.predict(L, inference_rule="kemeny"), Y)
    assert weighted < unweighted


@pytest.mark.parametrize("rule", ["kemeny", "weighted_kemeny", "kemeny_medoid", "weighted_kemeny_medoid",
                                  "pairwise_majority", "weighted_pairwise_majority", "position_estimation",
                                  "weighted_position_estimation"])
def test_all_rules_return_full_rankings(data, rule):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru, inference_rule=rule).fit(L[:50])
    out = lm.predict(L[:50])
    assert len(out) == 50 and all(sorted(y.permutation) == list(range(6)) for y in out)
    assert not any(y.is_partial for y in out)


def test_predict_does_not_alias_lf_outputs(data):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru, inference_rule="kemeny_medoid").fit(L[:20])
    out = lm.predict(L[:20])
    assert all(o is not lf for o, row in zip(out, L[:20]) for lf in row)


def test_unknown_rule_raises(data):
    ru, *_ = data
    with pytest.raises(ValueError):
        RankingLabelModel(ru, inference_rule="borda")
    with pytest.raises(ValueError):
        RankingLabelModel(ru, train_method="em")


def test_partial_rankings(data):
    ru, Y, thetas, _ = data
    rng = np.random.default_rng(5)
    L = sample_mallows_LFs(Y[:200], len(thetas), thetas, p=0.6, rng=rng)
    assert any(r.is_partial for row in L for r in row)
    lm = RankingLabelModel(ru, inference_rule="weighted_pairwise_majority").fit(L)
    assert np.all(np.isfinite(lm.thetas))
    assert ru.mean_kt_distance(lm.predict(L), Y[:200]) < 0.3


def test_backwards_compatible_api(data):
    ru, Y, thetas, L = data
    wsr = WeakSupRanking(ru)
    wsr.train({"train_method": "triplet_opt", "inference_rule": "weighted_kemeny"}, L)
    out = wsr.infer_ranking({"inference_rule": "weighted_kemeny"}, L)
    assert len(out) == len(L)


def test_numpy_input(data):
    ru, Y, thetas, L = data
    arr = np.array([[r.permutation for r in row] for row in L[:30]])
    lm = RankingLabelModel(ru).fit(arr)
    assert len(lm.predict(arr)) == 30


def test_save_load(tmp_path, data):
    ru, Y, thetas, L = data
    lm = RankingLabelModel(ru).fit(L[:30])
    lm.save(tmp_path / "m.pkl")
    assert np.allclose(RankingLabelModel.load(tmp_path / "m.pkl").thetas, lm.thetas)
