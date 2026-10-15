"""Synthetic data for the Gaussian (regression) label model."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal


def generate_synthetic_xy(n: int = 1000, d: int = 10, rng: Optional[np.random.Generator] = None):
    """``n`` points ``X ~ N(0, diag(U[0,1]^d))`` with ``Y = X @ 1``. Returns ``X, Y, var_Y``."""
    rng = np.random.default_rng() if rng is None else rng
    X_sigma = np.diag(rng.random(d))
    X = rng.multivariate_normal(np.zeros(d), X_sigma, size=n)
    beta = np.ones(d)
    return X, X @ beta, float(np.inner(beta, np.diag(X_sigma)))


def generate_lfs(m: int, n: int, Y, Y_var: float, seed: Optional[int] = 42, diag_fact: float = 5.0,
                 K_fact: float = -3.0):
    """Sample ``m`` LFs whose joint law with ``Y`` is Gaussian with a star-shaped precision matrix
    (LFs conditionally independent given ``Y``), rescaled so that ``Var(Y) = Y_var``.

    Returns ``L`` (n x m) and the joint covariance ``Sigma`` ((m+1) x (m+1), ``Y`` last).
    """
    K = np.zeros([m + 1, m + 1])
    K[:, m] = K[m, :] = K_fact
    K[m, m] = 0.0
    K += np.eye(m + 1) * diag_fact

    # adjust K[m, m] (Sherman-Morrison) so that the marginal variance of Y matches Y_var
    Sigma_temp = np.linalg.inv(K)
    Y_var_temp = Sigma_temp[m, m]
    un = np.zeros(m + 1)
    un[m] = 1.0
    ent = Sigma_temp @ np.outer(un, un) @ Sigma_temp
    alpha = (Y_var_temp - Y_var) / (ent[m, m] + (Y_var - Y_var_temp) * un @ Sigma_temp @ un)
    K_adj = K + alpha * np.outer(un, un)
    try:
        np.linalg.cholesky(K_adj)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"No valid LF covariance for m={m}, Y_var={Y_var}, diag_fact={diag_fact}, "
                         f"K_fact={K_fact}: the adjusted precision matrix is not positive definite. "
                         "Increase diag_fact or decrease |K_fact|.") from e
    Sigma = np.linalg.inv(K_adj)

    # LF | Y ~ N(Sigma_LY / Sigma_YY * Y, Sigma_LL - Sigma_LY Sigma_YL / Sigma_YY)
    Sig_bar = Sigma[:m, :m] - np.outer(Sigma[m, :m], Sigma[m, :m]) / Sigma[m, m]
    L = np.atleast_2d(multivariate_normal(mean=np.zeros(m), cov=Sig_bar, seed=seed).rvs(n))
    L = L + (np.asarray(Y)[:, None] * Sigma[:m, m][None, :] / Sigma[m, m])
    return L, Sigma
