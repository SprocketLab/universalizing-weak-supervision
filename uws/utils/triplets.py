"""Triplet estimators shared by every UWS label model.

All label models in UWS rely on the same identity: if labeling functions (LFs) ``a`` and ``b`` are
conditionally independent given the latent label ``y``, then (exactly for squared Euclidean distances,
approximately for many other metrics)

.. math:: \\mathbb{E}[d(\\lambda_a, \\lambda_b)] \\approx \\mathbb{E}[d(\\lambda_a, y)] + \\mathbb{E}[d(\\lambda_b, y)].

Given the observable matrix ``E[a, b]`` of mean LF-LF distances, the unobservable per-LF expected
distances ``e[a] = E[d(lambda_a, y)]`` are recovered from triplets ``(a, b, c)``:
``e[a] = (E[a,b] + E[a,c] - E[b,c]) / 2``.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np

TRIPLET_METHODS = ("triplet", "median_triplet", "mean_triplet", "lstsq")


def solve_triplet(e_ab: float, e_bc: float, e_ca: float) -> tuple[float, float, float]:
    """Solve ``x+y = e_ab, y+z = e_bc, z+x = e_ca`` and return ``(x, y, z)``."""
    t = (e_ab + e_bc + e_ca) / 2.0
    return t - e_bc, t - e_ca, t - e_ab


def order_by_max_distance(E: np.ndarray) -> list[int]:
    """Order LFs by decreasing pairwise distance (the ordering used by the original 'triplet' method)."""
    m = E.shape[0]
    pairs = sorted(((i, j, E[i, j]) for i in range(m) for j in range(i)), key=lambda x: x[2], reverse=True)
    order: list[int] = []
    for i, j, _ in pairs:
        for k in (i, j):
            if k not in order:
                order.append(k)
    return order


def _valid_triplets(m: int, groups: Optional[Sequence]) -> dict[int, list[tuple[int, int]]]:
    """For each LF a, the (b, c) partners usable in a triplet. With ``groups``, partners must come from
    groups other than a's (a diagnostic for correlated LF families)."""
    out = {}
    for a in range(m):
        others = [b for b in range(m) if b != a and (groups is None or groups[b] != groups[a])]
        out[a] = list(itertools.combinations(others, 2))
    return out


def individual_distances(
    E: np.ndarray,
    method: str = "median_triplet",
    groups: Optional[Sequence] = None,
    eps: float = 1e-6,
    clip: bool = True,
) -> np.ndarray:
    """Recover per-LF expected distances to the latent label from mean pairwise LF distances.

    Parameters
    ----------
    E : (m, m) array
        Symmetric matrix of mean distances between LF outputs.
    method : {"triplet", "median_triplet", "mean_triplet", "lstsq"}
        * ``triplet``: disjoint triplets over LFs ordered by decreasing distance (original method);
          leftover LFs (when ``m % 3 != 0``) are solved with two already-used LFs.
        * ``median_triplet`` / ``mean_triplet``: every triplet containing the LF, aggregated.
        * ``lstsq``: non-negative least squares over all pairs, ``min_e sum_{a<b} (E_ab - e_a - e_b)^2``.
    groups : sequence of length m, optional
        Family label per LF. Triplets then only combine LFs from different families
        (not used by ``triplet``/``lstsq``).
    eps : float
        Lower clip for the returned distances (so that ``1/e`` stays finite).
    clip : bool
        Whether to clip to ``eps``; disable to inspect raw (possibly negative) estimates.
    """
    E = np.asarray(E, dtype=float)
    m = E.shape[0]
    if E.shape != (m, m):
        raise ValueError(f"E must be square, got shape {E.shape}")
    if m < 3:
        raise ValueError(f"Triplet estimation needs at least 3 labeling functions, got {m}")
    if not np.all(np.isfinite(E)):
        raise ValueError("Pairwise LF distances contain inf/NaN (overflowing distances or LFs that never "
                         "co-vote); cannot estimate LF parameters")
    if method not in TRIPLET_METHODS:
        raise ValueError(f"Unknown triplet method {method!r}; choose from {TRIPLET_METHODS}")

    if method == "triplet":
        e = np.full(m, np.nan)
        order = order_by_max_distance(E)
        i = 0
        while i + 2 < m:
            a, b, c = order[i : i + 3]
            e[a], e[b], e[c] = solve_triplet(E[a, b], E[b, c], E[c, a])
            i += 3
        # leftovers: pair them with the first two LFs of the ordering
        for a in order[i:]:
            b, c = [x for x in order if x != a][:2]
            e[a] = solve_triplet(E[a, b], E[b, c], E[c, a])[0]
    elif method in ("median_triplet", "mean_triplet"):
        agg = np.median if method == "median_triplet" else np.mean
        partners = _valid_triplets(m, groups)
        e = np.empty(m)
        for a in range(m):
            if not partners[a]:
                raise ValueError(f"No valid triplet for LF {a} (check `groups`)")
            vals = [(E[a, b] + E[a, c] - E[b, c]) / 2.0 for b, c in partners[a]]
            e[a] = agg(vals)
    else:  # lstsq
        from scipy.optimize import nnls

        rows, rhs = [], []
        for a, b in itertools.combinations(range(m), 2):
            r = np.zeros(m)
            r[a] = r[b] = 1.0
            rows.append(r)
            rhs.append(E[a, b])
        e, _ = nnls(np.array(rows), np.array(rhs))

    if clip:
        e = np.maximum(e, eps)
    return e
