import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uws.end_model import RankingEndModel  # noqa: E402
from uws.end_model.ranking import listmle_loss, ndcg_at_k  # noqa: E402
from uws.ranking import Ranking, RankingUtils  # noqa: E402


def plackett_luce_nll(s, order):
    s = np.asarray(s, dtype=float)[list(order)]
    return -sum(s[i] - np.log(np.exp(s[i:]).sum()) for i in range(len(s)))


def test_listmle_matches_plackett_luce():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(3, 4))
    perms = np.array([rng.permutation(4) for _ in range(3)])
    observed = np.ones_like(perms, dtype=bool)
    got = float(listmle_loss(torch.tensor(scores), torch.tensor(perms), torch.tensor(observed)))
    want = np.mean([plackett_luce_nll(s, p) / 4 for s, p in zip(scores, perms)])
    assert got == pytest.approx(want, abs=1e-10)
    # Plackett-Luce probabilities of all orders sum to one
    total = sum(np.exp(-plackett_luce_nll(scores[0], p)) for p in itertools.permutations(range(4)))
    assert total == pytest.approx(1.0)


def test_listmle_ignores_masked_items():
    scores = torch.tensor([[0.3, -1.0, 2.0, 0.5]], dtype=torch.float64)
    perms = torch.tensor([[2, 0, 3, 1]])
    observed = torch.tensor([[True, True, False, False]])
    got = float(listmle_loss(scores, perms, observed))
    want = plackett_luce_nll(scores[0, [2, 0]].numpy(), [0, 1]) / 2
    assert got == pytest.approx(want, abs=1e-10)


def make_data(n, d=5, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d, f)).astype(np.float32)
    true_scores = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * rng.normal(size=(n, d))
    r_utils = RankingUtils(d)
    Y = [Ranking(np.argsort(-s), r_utils) for s in true_scores]
    return X, Y


def test_end_model_learns_rankings():
    X, Y = make_data(1200)
    em = RankingEndModel(n_features=4, epochs=15, seed=0).fit(X[:1000], Y[:1000])
    assert len(em.losses) == 15 and em.losses[-1] < em.losses[0]
    Y_pred = em.predict(X[1000:])
    assert len(Y_pred) == 200 and isinstance(Y_pred[0], Ranking)
    assert em.score(X[1000:], Y[1000:]) < 0.15  # random rankings score 0.5
    assert em.score(X[1000:], Y[1000:], metric="ndcg@3") > 0.9
    perms = np.array([y.permutation for y in Y[1000:]])
    assert em.score(X[1000:], perms) == pytest.approx(em.score(X[1000:], Y[1000:]))


def test_ndcg_perfect_and_reversed():
    perms = np.array([[0, 1, 2], [2, 1, 0]])
    perfect = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    assert ndcg_at_k(perfect, perms, 3) == pytest.approx(1.0)
    assert ndcg_at_k(-perfect, perms, 3) < 0.6


def test_end_model_validates_inputs():
    X, Y = make_data(10)
    with pytest.raises(ValueError):
        RankingEndModel(n_features=3).fit(X, Y)
    with pytest.raises(RuntimeError):
        RankingEndModel(n_features=4).predict(X)
    em = RankingEndModel(n_features=4, epochs=1).fit(X, Y)
    with pytest.raises(ValueError):
        em.score(X, Y, metric="spearman")
