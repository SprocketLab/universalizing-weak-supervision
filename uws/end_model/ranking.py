"""A learning-to-rank end model: a feed-forward scorer trained with the ListMLE loss (Xia et al., 2008)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..ranking.ranking import Ranking, RankingUtils


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("RankingEndModel needs `pip install uws[l2r]` (torch)") from e
    return torch


def _as_permutations(Y, d: int) -> tuple[np.ndarray, np.ndarray]:
    """``(n, d)`` item orders (best first) and a boolean mask of observed positions."""
    if not isinstance(Y[0], Ranking):
        perms = np.asarray(Y, dtype=int)
        if perms.shape[1:] != (d,):
            raise ValueError(f"Expected rankings of {d} items, got shape {perms.shape}")
        return perms, np.ones_like(perms, dtype=bool)
    perms = np.array([y.permutation for y in Y], dtype=int)
    observed = np.array([[m == 0 for m in y.mask_permutation] for y in Y], dtype=bool)
    if perms.shape[1] != d:
        raise ValueError(f"Rankings have {perms.shape[1]} items but X has {d}")
    return perms, observed


def listmle_loss(scores, perms, observed):
    """Negative log-likelihood of the target orders under the Plackett-Luce model of ``scores``.

    ``scores`` is ``(n, d)`` (one score per item), ``perms`` gives each example's items best first and
    ``observed`` marks positions to keep: unobserved items of partial rankings are left out of the likelihood.
    Returns the mean over examples of the per-position average loss.
    """
    torch = _torch()

    s = torch.gather(scores, 1, perms)
    s = s.masked_fill(~observed, float("-inf"))
    tail_lse = torch.flip(torch.logcumsumexp(torch.flip(s, [1]), 1), [1])  # log sum_{k >= i} exp(s_k)
    terms = torch.where(observed, tail_lse - s, torch.zeros_like(s))
    return (terms.sum(1) / observed.sum(1).clamp(min=1)).mean()


class RankingEndModel:
    """Score each item from its features and rank items by score.

    The network is the one used in the paper's experiments: ``num_layers`` linear layers of width
    ``hidden_dim`` with ReLU, where the input features and every hidden layer are normalized over the items of
    each query (item set). It is trained with the ListMLE loss and SGD. Requires ``uws[l2r]`` (torch).

    The defaults train on 16 queries per step. ``batch_size=1, lr=0.01`` is the paper's exact schedule (one
    query per step); on the board-games data both reach the same test Kendall distance (about 0.16) and the
    defaults are about ten times faster.

    Parameters
    ----------
    n_features : int
        Number of features per item.
    hidden_dim, num_layers : int
        Width and depth of the scoring network.
    lr, weight_decay : float
        SGD learning rate and L2 penalty.
    epochs, batch_size : int
        Training schedule; ``batch_size`` counts queries.
    seed : int, optional
        Seed for weight initialization and batch order.

    Examples
    --------
    >>> em = RankingEndModel(n_features=X_train.shape[2]).fit(X_train, Y_train)   # doctest: +SKIP
    >>> Y_pred = em.predict(X_test)                                                 # doctest: +SKIP
    >>> em.score(X_test, Y_test)                                                    # doctest: +SKIP
    """

    def __init__(self, n_features: int, hidden_dim: int = 30, num_layers: int = 3, lr: float = 0.05,
                 weight_decay: float = 1e-4, epochs: int = 8, batch_size: int = 16, seed: Optional[int] = None):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.net = None
        self.losses: list[float] = []

    def _build(self):
        torch = _torch()
        nn = torch.nn

        class QueryNorm(nn.Module):
            """Normalize each feature over the items of one query (item set), with a learned scale and shift."""

            def __init__(self, dim: int, eps: float = 1e-5):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim))
                self.eps = eps

            def forward(self, x):  # (n, d, dim)
                mean = x.mean(1, keepdim=True)
                var = x.var(1, unbiased=False, keepdim=True)
                return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

        if self.seed is not None:
            torch.manual_seed(self.seed)
        layers: list = [QueryNorm(self.n_features)]
        width = self.n_features
        for _ in range(self.num_layers - 1):
            lin = nn.Linear(width, self.hidden_dim)
            nn.init.xavier_normal_(lin.weight)
            layers += [lin, QueryNorm(self.hidden_dim), nn.ReLU()]
            width = self.hidden_dim
        out = nn.Linear(width, 1)
        nn.init.xavier_normal_(out.weight)
        layers.append(out)
        return nn.Sequential(*layers)

    def _scores(self, X):
        return self.net(X).squeeze(-1)

    def fit(self, X, Y) -> RankingEndModel:
        """Train on item features ``X`` of shape ``(n, d, n_features)`` and target rankings ``Y``.

        ``Y`` is a list of :class:`~uws.ranking.Ranking` (for example a label model's output; masked items of
        partial rankings are ignored) or an ``(n, d)`` array of item indices, best first.
        """
        torch = _torch()

        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if X.ndim != 3 or X.shape[2] != self.n_features:
            raise ValueError(f"X must have shape (n, d, {self.n_features}), got {tuple(X.shape)}")
        perms, observed = _as_permutations(Y, X.shape[1])
        perms, observed = torch.as_tensor(perms), torch.as_tensor(observed)
        self.net = self._build()
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        self.losses = []
        self.net.train()
        for _ in range(self.epochs):
            total = 0.0
            for idx in torch.randperm(len(X), generator=gen).split(self.batch_size):
                loss = listmle_loss(self._scores(X[idx]), perms[idx], observed[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.losses.append(total / len(X))
        self.net.eval()
        return self

    def predict_scores(self, X) -> np.ndarray:
        """Item scores, shape ``(n, d)``; higher means ranked higher."""
        torch = _torch()

        if self.net is None:
            raise RuntimeError("Call fit() first")
        with torch.no_grad():
            return self._scores(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

    def predict(self, X) -> list[Ranking]:
        """Predicted rankings (items sorted by decreasing score)."""
        scores = self.predict_scores(X)
        r_utils = RankingUtils(scores.shape[1])
        return [Ranking(np.argsort(-s, kind="stable"), r_utils) for s in scores]

    def score(self, X, Y, metric: str = "kendall_tau") -> float:
        """Evaluate on ``(X, Y)``.

        ``metric`` is ``"kendall_tau"`` (mean normalized Kendall tau distance, lower is better) or ``"ndcg@k"``
        (NDCG of the top ``k`` items with relevance ``d - 1 - true position``, higher is better).
        """
        d = np.asarray(X).shape[1]
        perms, _ = _as_permutations(Y, d)
        if metric == "kendall_tau":
            r_utils = RankingUtils(d)
            Y_true = Y if isinstance(Y[0], Ranking) else [Ranking(p, r_utils) for p in perms]
            return r_utils.mean_kt_distance(self.predict(X), Y_true)
        if metric.startswith("ndcg@"):
            return ndcg_at_k(self.predict_scores(X), perms, int(metric[5:]))
        raise ValueError(f"Unknown metric {metric!r}; use 'kendall_tau' or 'ndcg@k'")


def ndcg_at_k(scores: np.ndarray, perms: np.ndarray, k: int) -> float:
    """Mean NDCG@k of ``scores`` against true orders ``perms`` (relevance ``d - 1 - position``)."""
    n, d = perms.shape
    k = min(k, d)
    rel = np.empty((n, d))
    rel[np.arange(n)[:, None], perms] = d - 1 - np.arange(d)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    dcg = ((2.0 ** np.take_along_axis(rel, top, 1) - 1) * discount).sum(1)
    ideal = ((2.0 ** (d - 1 - np.arange(k)) - 1) * discount).sum()
    return float(np.mean(dcg / ideal))


__all__ = ["RankingEndModel", "listmle_loss", "ndcg_at_k"]
