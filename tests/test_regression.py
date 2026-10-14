import warnings

import numpy as np
import pytest

from uws.regression import ContinuousLabelModel, generate_lfs, generate_synthetic_xy


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X, Y, v = generate_synthetic_xy(3000, 10, rng)
    L, Sigma = generate_lfs(6, 3000, Y, v, seed=0)
    return X, Y, v, L, Sigma


def oracle_mse(Y, L, Sigma):
    m = L.shape[1]
    return np.mean((Y - Sigma[m, :m] @ np.linalg.solve(Sigma[:m, :m], L.T)) ** 2)


@pytest.mark.parametrize("kw", [{}, {"solver": "empirical"}, {"estimator": "lstsq"}, {"estimator": "triplet"},
                                {"standardize": True}, {"legacy": True}])
def test_close_to_oracle(data, kw):
    _, Y, v, L, Sigma = data
    lm = ContinuousLabelModel(**kw).fit(L[:2000], v)
    assert lm.score(Y[2000:], L[2000:]) < 1.1 * oracle_mse(Y[2000:], L[2000:], Sigma)


def test_recovers_accuracies(data):
    _, Y, v, L, Sigma = data
    lm = ContinuousLabelModel(center=False).fit(L, v)
    assert np.allclose(lm.accuracies, Sigma[:6, 6], rtol=0.1)


def test_predict_on_any_n(data):
    _, Y, v, L, _ = data
    lm = ContinuousLabelModel().fit(L[:100], v)
    assert lm.predict(L[:7]).shape == (7,)
    mean, var = lm.predict(L[:3], return_var=True)
    assert np.all(var > 0)


def test_legacy_predict_on_any_n(data):
    _, Y, v, L, _ = data
    lm = ContinuousLabelModel(legacy=True).fit(L[:100], v, median=False)
    assert lm.predict(L[:7]).shape == (7,)


def test_near_duplicate_and_mixed_scale_lfs_are_stable(data):
    _, Y, v, L, Sigma = data
    rng = np.random.default_rng(1)
    L2 = np.hstack([L, L[:, :3] + 1e-4 * rng.standard_normal((len(L), 3)), 100 * L[:, 3:4]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = ContinuousLabelModel().fit(L2[:2000], v).score(Y[2000:], L2[2000:])
    assert np.isfinite(new)
    assert new < 1.5 * oracle_mse(Y[2000:], L[2000:], Sigma)


def test_ill_conditioning_warns(data):
    _, Y, v, L, _ = data
    L2 = np.hstack([L, L[:, :1] + 1e-9])
    with pytest.warns(UserWarning, match="ill-conditioned"):
        ContinuousLabelModel().fit(L2, v)


def test_too_few_lfs(data):
    _, Y, v, L, _ = data
    with pytest.raises(ValueError):
        ContinuousLabelModel().fit(L[:, :2], v)


def test_score_metrics(data):
    _, Y, v, L, _ = data
    lm = ContinuousLabelModel().fit(L, v)
    lm.predict(L)
    assert lm.score(Y, metric="mae") > 0 and lm.score(Y, metric="r2") < 1
    with pytest.raises(ValueError):
        lm.score(Y, metric="huber")


def test_generate_lfs_rejects_invalid_parameters():
    Y = np.zeros(10)
    with pytest.raises(ValueError, match="positive definite"):
        generate_lfs(5, 10, Y, 1.0, diag_fact=-1.0)
