"""Geodesic regression ``y = exp_q(x alpha)`` fit by Riemannian gradient descent."""

from __future__ import annotations

from .synthetic import sample_params


class GeodesicRegressor:
    def __init__(self, man, dim: int):
        self.man = man
        self.dim = dim
        q, alpha = sample_params(man, dim, zero=True)
        self.q = q.clone().requires_grad_(True)
        self.alpha = alpha.clone().requires_grad_(True)

    def fit(self, X, Y, epochs: int = 300, lr: float = 0.01) -> float:
        import torch

        for _ in range(epochs):
            loss = torch.mean(self.man.dist2(Y, self.man.expmap(self.q, X * self.alpha)))
            loss.backward()
            with torch.no_grad():
                dq = self.man.egrad2rgrad(self.q, self.q.grad)
                q_new = self.man.expmap(self.q, -lr * dq)
                dalpha = self.man.egrad2rgrad(self.q, self.alpha.grad)
                alpha_new = self.man.transp(self.q, q_new, self.alpha - lr * dalpha)
                self.q.copy_(q_new)
                self.alpha.copy_(alpha_new)
            self.q.grad.zero_()
            self.alpha.grad.zero_()
        with torch.no_grad():
            return float(torch.mean(self.man.dist2(Y, self.predict(X))))

    def predict(self, X):
        import torch

        with torch.no_grad():
            return self.man.expmap(self.q, X * self.alpha)
