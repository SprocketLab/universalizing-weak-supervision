import numpy as np
import pytest

from uws.metric import MetricLabelModel, sample_metric_lfs


@pytest.fixture(scope="module")
def graph_metric():
    nx = pytest.importorskip("networkx")
    G = nx.gnm_random_graph(10, 20, seed=1)
    while not nx.is_connected(G):
        G = nx.gnm_random_graph(10, 20)
    return nx.floyd_warshall_numpy(G)


def test_label_model_beats_majority_vote(graph_metric):
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 10, 3000)
    thetas = np.array([0.4, 2.5] * 5)
    L = sample_metric_lfs(graph_metric, Y, thetas, rng=rng)
    lm = MetricLabelModel(graph_metric).fit(L)
    assert lm.score(L, Y)["accuracy"] > np.mean(lm.predict(L, vote=True) == Y) + 0.05
    assert np.all(lm.thetas[1::2] > lm.thetas[0::2])


def test_abstains_are_ignored(graph_metric):
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 10, 2000)
    L = sample_metric_lfs(graph_metric, Y, [1.0, 2.0, 1.5, 2.0], rng=rng)
    L[rng.random(L.shape) < 0.2] = -1
    lm = MetricLabelModel(graph_metric).fit(L)
    assert np.all(np.isfinite(lm.thetas))
    assert lm.predict_proba(L).shape == (2000, 10)


def test_callable_distance():
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 5, 300)
    D = np.abs(np.subtract.outer(np.arange(5), np.arange(5))).astype(float)
    L = sample_metric_lfs(D, Y, [1.0, 2.0, 3.0], rng=rng)
    a = MetricLabelModel(D).fit(L)
    b = MetricLabelModel(lambda x, y: abs(x - y), candidates=list(range(5))).fit(L)
    assert np.allclose(a.thetas, b.thetas)
    assert list(a.predict(L)) == list(b.predict(L))


def test_exact_theta(graph_metric):
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 10, 4000)
    L = sample_metric_lfs(graph_metric, Y, [0.5, 1.0, 2.0, 3.0], rng=rng)
    lm = MetricLabelModel(graph_metric, exact_theta=True).fit(L)
    assert np.all(np.diff(lm.thetas) > 0)
