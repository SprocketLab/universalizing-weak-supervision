"""Label model for manifold-valued labels (e.g. points on the hyperboloid), via geoopt."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..base import BaseLabelModel
from ..utils.triplets import individual_distances


def _require_geoopt():
    try:
        import geoopt  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("The geodesic label model needs `pip install uws[geodesic]` (torch, geoopt)") from e


class GeodesicLabelModel(BaseLabelModel):
    """LFs ``lambda_j ~ exp(-theta_j d^2(lambda_j, y))`` on a Riemannian manifold.

    Parameters
    ----------
    manifold : geoopt.Manifold, optional
        Defaults to ``geoopt.Lorentz()`` (hyperboloid model).
    method : str
        Triplet estimator for the expected squared distances.
    power : float
        Distance exponent used for the triplet estimates. The default 1 matches hyperbolic geometry,
        where geodesics between noisy LFs pass close to the label so distances are close to additive;
        use 2 for (near-)Euclidean manifolds. ``legacy=True`` uses the notebook's ``mean d / dim`` with
        disjoint consecutive triplets.
    legacy : bool
    num_triplets : int, optional
        Backwards compatibility with the notebook API (number of disjoint LF triplets).
    """

    def __init__(self, manifold=None, method: str = "median_triplet", power: float = 1.0, legacy: bool = False,
                 num_triplets: Optional[int] = None):
        _require_geoopt()
        if manifold is None:
            from geoopt.manifolds.lorentz import Lorentz

            manifold = Lorentz()
        self.man = manifold
        self.method = "triplet" if legacy else method
        self.power = 1.0 if legacy else power
        self.legacy = legacy
        self.num_triplets = num_triplets
        self.thetas = None

    def lf_distance_matrix(self, lambdas: Sequence) -> np.ndarray:
        import torch

        m = len(lambdas)
        dim = lambdas[0].shape[1]
        E = np.zeros((m, m))
        with torch.no_grad():
            for a in range(m):
                for b in range(a + 1, m):
                    # float64 + nan guard: acosh of inner products that round below 1 for nearby points
                    d = torch.nan_to_num(self.man.dist(lambdas[a].double(), lambdas[b].double()), nan=0.0)
                    d = d**self.power
                    E[a, b] = E[b, a] = float(d.mean()) / (dim if self.legacy else 1.0)
        return E

    def fit(self, lambdas: Sequence, groups: Optional[Sequence] = None) -> GeodesicLabelModel:
        """``lambdas``: list of ``m`` tensors ``(n, dim)`` of LF outputs."""
        if self.num_triplets is not None:
            lambdas = lambdas[: 3 * self.num_triplets]
        if self.legacy:
            # the notebook solved disjoint consecutive triplets (0,1,2), (3,4,5), ...
            e = []
            for i in range(len(lambdas) // 3):
                E = self.lf_distance_matrix(lambdas[3 * i : 3 * i + 3])
                ab, bc, ac = E[0, 1], E[1, 2], E[0, 2]
                e += list(np.linalg.solve([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [ab, bc, ac]))
            self.expected_dists = np.maximum(np.array(e), 1e-6)
        else:
            self.E_ = self.lf_distance_matrix(lambdas)
            self.expected_dists = individual_distances(self.E_, self.method, groups=groups)
        self.thetas = 1.0 / self.expected_dists
        return self

    def predict(self, lambdas: Sequence, max_steps: int = 200, lr: Optional[float] = None, vote: bool = False,
                init: str = "random", tol: float = 1e-6, verbose: bool = False):
        """Weighted Fréchet mean ``argmin_y sum_j theta_j d^2(lambda_j, y)`` for every example (batched),
        by Riemannian gradient descent.

        With ``lr=None`` (default) weights are normalized, computations run in float64 and each step uses
        a backtracking (Armijo) line search per example, so no step size needs tuning. A float ``lr``
        reproduces the notebook's fixed-step descent on the unnormalized objective (sensitive to the scale
        of theta). ``init="random"`` starts from a random point near the origin, as in the notebook, which is
        numerically the safest start on strongly curved hyperboloids; ``"extrinsic"`` starts from the
        projected weighted ambient mean of the LF outputs.
        """
        import torch

        m = len(self.thetas)
        dtype = lambdas[0].dtype
        lambdas = [l.double() for l in lambdas[:m]] if lr is None else list(lambdas[:m])
        w = np.ones(m) if vote else np.asarray(self.thetas, dtype=float)
        if lr is None:
            w = w / w.sum()
        wt = torch.as_tensor(w, dtype=lambdas[0].dtype)
        if init == "extrinsic":
            y = self.man.projx(sum(wt[j] / wt.sum() * lambdas[j] for j in range(m)))
        else:
            y = self.man.projx(torch.randn_like(lambdas[0]))

        def objective(y_):
            return sum(wt[j] * self.man.dist2(lambdas[j], y_) for j in range(m))  # (n,)

        step = torch.ones(len(y), dtype=y.dtype)
        for it in range(max_steps):
            y = y.detach().requires_grad_(True)
            f = objective(y)
            (grad,) = torch.autograd.grad(f.sum(), y)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)  # d acosh at coincident points
            with torch.no_grad():
                rgrad = self.man.egrad2rgrad(y, grad)
                if lr is not None:
                    y_new = self.man.expmap(y, -lr * rgrad)
                else:
                    gnorm2 = self.man.inner(y, rgrad, rgrad).reshape(-1).clamp_min(0)
                    step = torch.clamp(step * 2, max=1.0)
                    for _ in range(30):  # backtrack until sufficient decrease for every example
                        y_new = self.man.expmap(y, -step[:, None] * rgrad)
                        f_new = objective(y_new)
                        bad = ~(f_new <= f - 1e-4 * step * gnorm2) & (step > 1e-12)
                        if not bad.any():
                            break
                        step = torch.where(bad, step / 2, step)
                    ok = torch.isfinite(f_new) & (f_new <= f)
                    y_new = torch.where(ok[:, None], y_new, y)
                rel = float(((f - objective(y_new)).abs() / f.abs().clamp_min(1.0)).max())
            y = y_new.detach()
            if verbose:
                print(it, float(f.sum()), rel)
            if rel < tol:
                break
        return y.detach().to(dtype)
