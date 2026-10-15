"""Gaussian label model for real-valued labels (regression)."""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Optional, Union

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .base import BaseLabelModel

ESTIMATORS = ("triplet", "median_triplet", "lstsq")


def _ledoit_wolf_alpha(X: np.ndarray) -> float:
    from sklearn.covariance import ledoit_wolf_shrinkage

    return float(ledoit_wolf_shrinkage(X, assume_centered=True))


class ContinuousLabelModel(BaseLabelModel):
    """Jointly Gaussian ``(lambda_1..lambda_m, Y)`` with LFs conditionally independent given ``Y``.

    Then ``Cov(lambda_i, lambda_j) = a_i a_j / var_Y`` for ``i != j`` with ``a_i = Cov(lambda_i, Y)``,
    which identifies the accuracies ``a`` from the observable LF covariance ``C`` (triplets:
    ``a_i^2 = C_ij C_ik var_Y / C_jk``). Prediction is the posterior mean
    ``E[Y | lambda] = mu_Y + a^T C^{-1} (lambda - mu_lambda)``.

    Numerical safeguards (all on by default; ``legacy=True`` restores the paper's implementation):

    * no explicit inverse. With ``solver="structured"`` (default) the LF covariance implied by the model,
      ``a a^T / var_Y + diag(s^2)``, is used, whose inverse is closed-form (Sherman-Morrison): the
      posterior is always well defined, however correlated or numerous the LFs. With
      ``solver="empirical"`` the sample covariance is used through a Cholesky solve after shrinkage towards
      ``diag(C)`` (Ledoit-Wolf intensity by default, raised automatically if the joint covariance would
      not be PSD);
    * centering (and optional standardization) of the LFs;
    * robust accuracy estimation: degenerate triplets (``|C_jk|`` tiny) are skipped, medians are taken in
      the log domain, signs are recovered from ``C`` and ``|a_i| <= sqrt(C_ii var_Y)`` is enforced;

    Parameters
    ----------
    estimator : {"median_triplet", "triplet", "lstsq"}
        ``lstsq`` fits all pairs jointly (``min_a sum_{i<j} (C_ij - a_i a_j / var_Y)^2``).
    solver : {"structured", "empirical"}
        Which LF covariance the posterior uses (see above).
    shrinkage : "auto", float or None
        ``solver="empirical"`` only: shrinkage intensity in ``[0, 1]``; ``"auto"`` uses the
        Ledoit-Wolf estimate, ``0``/``None`` disables it.
    standardize : bool
        Z-score LFs before fitting (scale-invariant conditioning).
    center : bool
        Estimate and remove LF means. Disable when LFs and ``Y`` are known to be zero-mean (then second
        moments are used directly and ``mean_Y`` defaults to 0).
    legacy : bool
        Reproduce the original implementation (uncentered second moments, explicit inverse, random
        triplets or median, no safeguards).
    eps : float
        Threshold below which ``|C_jk|`` makes a triplet degenerate.
    """

    def __init__(self, estimator: str = "median_triplet", solver: str = "structured",
                 shrinkage: Union[str, float, None] = "auto", standardize: bool = False, legacy: bool = False,
                 eps: float = 1e-8, center: bool = True, use_triplets: bool = True):
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator {estimator!r}; choose from {ESTIMATORS}")
        if solver not in ("structured", "empirical"):
            raise ValueError(f"Unknown solver {solver!r}")
        self.solver = solver
        self.center = center
        self.estimator = estimator
        self.shrinkage = shrinkage
        self.standardize = standardize
        self.legacy = legacy
        self.eps = eps
        self.use_triplets = use_triplets  # kept for backwards compatibility
        self.thetas = None

    # ------------------------------------------------------------------ fitting
    def fit(self, L_train, var_Y: float, mean_Y: Optional[float] = None, median: Optional[bool] = None,
            seed: int = 10) -> ContinuousLabelModel:
        """Estimate LF accuracies from unlabeled LF outputs ``L_train`` (n x m).

        ``var_Y`` (and optionally ``mean_Y``) of the latent label must be supplied: the model is only
        identified up to the scale of ``Y``. ``mean_Y`` defaults to the least-squares fit of the LF means
        assuming unbiased LFs (``E[lambda_i] = a_i mu_Y / var_Y``). ``median`` is a
        backwards-compatible alias for ``estimator`` (True -> median_triplet, False -> triplet).
        """
        L = np.asarray(L_train, dtype=float)
        if L.ndim != 2:
            raise ValueError(f"L_train must be 2-D (n x m), got shape {L.shape}")
        self.n, self.m = n, m = L.shape
        if m < 3:
            raise ValueError(f"At least 3 LFs are needed to identify accuracies, got {m}")
        if var_Y <= 0:
            raise ValueError("var_Y must be positive")
        estimator = self.estimator if median is None else ("median_triplet" if median else "triplet")
        self.var_Y = float(var_Y)
        if self.legacy:
            return self._fit_legacy(L, estimator == "median_triplet", seed)

        self.mu_L = L.mean(0) if self.center else np.zeros(m)
        X = L - self.mu_L
        self.scale_L = X.std(0) if self.standardize else np.ones(m)
        self.scale_L = np.where(self.scale_L > 0, self.scale_L, 1.0)
        X = X / self.scale_L
        C = X.T @ X / n
        self.O = C
        sd0 = np.sqrt(np.diag(C))
        self.cond_ = float(np.linalg.cond(C / np.outer(sd0, sd0))) if np.all(sd0 > 0) else np.inf
        if self.cond_ > 1e8:
            warnings.warn(f"LF covariance is ill-conditioned (cond={self.cond_:.2e}); shrinkage is applied "
                          "to the solve, but accuracies of near-duplicate LFs are poorly identified.",
                          stacklevel=2)

        a = self._accuracies(C, estimator)
        a = np.clip(a, -np.sqrt(np.diag(C) * self.var_Y), np.sqrt(np.diag(C) * self.var_Y))

        self.shrinkage_ = 0.0
        if self.solver == "structured":
            # C = a a^T / var_Y + diag(s2) under the model: closed-form posterior, no inversion
            s2 = np.maximum(np.diag(C) - a**2 / self.var_Y, self.eps * np.maximum(np.diag(C), self.eps))
            t = a / s2
            denom = 1.0 + float(a @ t) / self.var_Y
            w = t / denom
            explained = float(a @ w)
            self.noise_var_ = s2
        else:
            # shrink the correlation matrix towards I (C towards diag(C)), scale-aware
            sd = np.sqrt(np.diag(C))
            sd = np.where(sd > 0, sd, 1.0)
            alpha = self._shrinkage_alpha(X / sd)
            w = self._solve((1 - alpha) * C + alpha * np.diag(np.diag(C)), a)
            explained = float(a @ w)
            if explained >= self.var_Y:
                warnings.warn("The joint covariance implied by the estimates is not PSD (negative posterior "
                              "variance); consider solver='structured'.", stacklevel=2)
            self.shrinkage_ = alpha
        if mean_Y is None:
            # E[lambda_i] = (a_i / var_Y) mu_Y for unbiased LFs: least-squares estimate of mu_Y
            b = a / self.var_Y / self.scale_L
            mean_Y = float(b @ self.mu_L / (b @ b)) if self.center else 0.0
        self.mean_Y = float(mean_Y)
        self.accuracies = a
        self.weights_ = w
        self.posterior_var_ = self.var_Y - explained
        self.thetas = a / np.sqrt(np.diag(C) * self.var_Y)  # correlation of each LF with Y
        self.Sigma_hat = np.block([[C, a[:, None]], [a[None, :], np.array([[self.var_Y]])]])
        return self

    def _shrinkage_alpha(self, X: np.ndarray) -> float:
        if self.shrinkage in (None, 0, 0.0):
            return 0.0
        if self.shrinkage == "auto":
            return _ledoit_wolf_alpha(X)
        alpha = float(self.shrinkage)
        if not 0 <= alpha <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        return alpha

    @staticmethod
    def _solve(C: np.ndarray, a: np.ndarray) -> np.ndarray:
        try:
            return cho_solve(cho_factor(C), a)
        except LinAlgError:
            return np.linalg.lstsq(C, a, rcond=None)[0]

    def _signs(self, C: np.ndarray) -> np.ndarray:
        """Signs of the accuracies from the sign pattern of C (leading eigenvector), mostly positive."""
        S = np.sign(C)
        np.fill_diagonal(S, 0)
        _, vecs = np.linalg.eigh(S)
        s = np.sign(vecs[:, -1])
        s[s == 0] = 1
        return s if s.sum() >= 0 else -s

    def _accuracies(self, C: np.ndarray, estimator: str) -> np.ndarray:
        m, v = self.m, self.var_Y
        absC = np.abs(C)
        log_mag = {i: [] for i in range(m)}
        degenerate = 0
        for i, j, k in itertools.combinations(range(m), 3):
            for x, y, z in ((i, j, k), (j, i, k), (k, i, j)):
                if min(absC[x, y], absC[x, z], absC[y, z]) < self.eps:
                    degenerate += 1
                    continue
                log_mag[x].append(0.5 * (np.log(absC[x, y]) + np.log(absC[x, z]) - np.log(absC[y, z]) + np.log(v)))
        if degenerate:
            warnings.warn(f"{degenerate} degenerate triplet estimates (|C_jk| < {self.eps}) were skipped",
                          stacklevel=3)
        if any(len(vals) == 0 for vals in log_mag.values()):
            raise ValueError("Some LF has no non-degenerate triplet; LFs look uncorrelated with each other")
        if estimator == "triplet":
            mag = np.array([np.exp(log_mag[i][0]) for i in range(m)])
        else:
            mag = np.array([np.exp(np.median(log_mag[i])) for i in range(m)])
        a = self._signs(C) * mag
        if estimator == "lstsq":
            a = self._lstsq_refine(C, a)
        return a

    def _lstsq_refine(self, C: np.ndarray, a0: np.ndarray, n_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
        """Coordinate descent for ``min_b sum_{i != j} (C_ij - b_i b_j)^2`` with ``b = a / sqrt(var_Y)``."""
        b = a0 / np.sqrt(self.var_Y)
        off = C - np.diag(np.diag(C))
        for _ in range(n_iter):
            b_old = b.copy()
            for i in range(self.m):
                others = np.arange(self.m) != i
                denom = (b[others] ** 2).sum()
                if denom > 0:
                    b[i] = (off[i, others] @ b[others]) / denom
            if np.max(np.abs(b - b_old)) < tol:
                break
        if b.sum() < 0:
            b = -b
        return b * np.sqrt(self.var_Y)

    def _fit_legacy(self, L: np.ndarray, median: bool, seed: int):
        n, m = L.shape
        self.O = L.T @ L / n
        self.Sigma_hat = np.zeros([m + 1, m + 1])
        self.Sigma_hat[:m, :m] = self.O
        rnd = random.Random(seed)
        if median:
            acc = {i: [] for i in range(m)}
            for i, j, k in itertools.combinations(range(m), 3):
                acc[i].append(np.sqrt(self.O[i, j] * self.O[i, k] * self.var_Y / self.O[j, k]))
                acc[j].append(np.sqrt(self.O[j, i] * self.O[j, k] * self.var_Y / self.O[i, k]))
                acc[k].append(np.sqrt(self.O[k, i] * self.O[k, j] * self.var_Y / self.O[i, j]))
            a = np.array([np.median(acc[i]) for i in range(m)])
        else:
            a = np.zeros(m)
            for i in range(m):
                j, k = rnd.sample(sorted(set(range(m)) - {i}), 2)
                a[i] = np.sqrt(self.O[i, j] * self.O[i, k] * self.var_Y / self.O[j, k])
        self.Sigma_hat[:m, m] = self.Sigma_hat[m, :m] = a
        self.Sigma_hat[m, m] = self.var_Y
        self.accuracies = a
        self.mu_L, self.mean_Y, self.scale_L = np.zeros(m), 0.0, np.ones(m)
        self.weights_ = np.linalg.inv(self.O) @ a
        self.posterior_var_ = self.var_Y - float(a @ self.weights_)
        self.thetas = a
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, L, return_var: bool = False):
        """Posterior mean of ``Y`` for each row of ``L`` (any number of rows)."""
        if self.thetas is None:
            raise RuntimeError("Call fit() first")
        L = np.asarray(L, dtype=float)[:, : self.m]
        self.Y_hat = self.mean_Y + ((L - self.mu_L) / self.scale_L) @ self.weights_
        if return_var:
            return self.Y_hat, np.full(len(L), max(self.posterior_var_, 0.0))
        return self.Y_hat

    inference = predict  # backwards compatibility (old name in ws_regression.LabelModel)

    def score(self, Y, L=None, metric: str = "mse") -> float:
        """Error of the predictions on ``L`` (default: the last ``predict`` call) against ``Y``."""
        Y_hat = self.predict(L) if L is not None else self.Y_hat
        Y = np.asarray(Y, dtype=float)
        if metric == "mse":
            return float(np.mean((Y - Y_hat) ** 2))
        if metric == "mae":
            return float(np.mean(np.abs(Y - Y_hat)))
        if metric == "r2":
            return float(1 - np.sum((Y - Y_hat) ** 2) / np.sum((Y - Y.mean()) ** 2))
        raise ValueError(f"Unknown metric {metric!r}")


LabelModel = ContinuousLabelModel
