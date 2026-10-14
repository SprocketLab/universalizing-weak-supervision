"""Synthetic ranking LFs sampled from Mallows models, with optional masking (partial rankings)."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Optional, Sequence

import numpy as np

from .mallows import Mallows
from .ranking import Ranking, RankingUtils


def generate_true_rankings(d: int, n: int, rng: Optional[np.random.Generator] = None,
                           r_utils: Optional[RankingUtils] = None) -> list[Ranking]:
    """``n`` uniformly random rankings over ``d`` items."""
    rng = np.random.default_rng() if rng is None else rng
    r_utils = RankingUtils(d) if r_utils is None else r_utils
    return [r_utils.get_random_ranking(rng) for _ in range(n)]


def get_mask(y: Ranking, m: int, p: float, rng: Optional[np.random.Generator] = None) -> list[list[int]]:
    """Per-LF lists of items to mask: each item is hidden from ``m - ceil(p m)`` random LFs, and every LF
    keeps at least two items.

    ``p`` is the fraction of LFs that observe each item.
    """
    rng = np.random.default_rng() if rng is None else rng
    d = len(y)
    k = m - ceil(p * m)
    masks: list[list[int]] = [[] for _ in range(m)]
    for x in y.permutation:
        for lf in rng.choice(m, size=k, replace=False):
            masks[lf].append(x)
    for i in range(m):
        if len(masks[i]) > d - 2:
            masks[i] = masks[i][2:]
    return masks


def apply_masks(Y, L, m: int, p: Optional[float] = None, d_mask: Optional[int] = None,
                rng: Optional[np.random.Generator] = None) -> None:
    """Mask LF outputs in place: randomly with observation probability ``p`` or, if ``p`` is None, keep
    only the top ``d_mask`` items of every LF (mask all others)."""
    rng = np.random.default_rng() if rng is None else rng
    for i in range(len(Y)):
        if p is not None:
            for j, mask in enumerate(get_mask(Y[i], m, p, rng)):
                if mask:
                    L[i][j].mask_items(mask, rng)
        else:
            for j in range(m):
                L[i][j].mask_items(L[i][j].permutation[d_mask:], rng)


def sample_mallows_LFs(Y: Sequence[Ranking], m: int, thetas, p: Optional[float] = None,
                       d_mask: Optional[int] = None, sampler: str = "rim",
                       rng: Optional[np.random.Generator] = None) -> list[list[Ranking]]:
    """Sample ``m`` Mallows LFs (one theta each) around every true ranking in ``Y``."""
    rng = np.random.default_rng() if rng is None else rng
    d = len(Y[0])
    r_utils = Y[0].r_utils if isinstance(Y[0], Ranking) else RankingUtils(d)
    lst_mlw = [Mallows(r_utils, theta, sampler=sampler, rng=rng) for theta in np.asarray(thetas)]
    L = [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y]
    if p is not None and p >= 0.05:
        apply_masks(Y, L, m, p, rng=rng)
    elif d_mask is not None and d_mask > 0:
        apply_masks(Y, L, m, None, d_mask, rng=rng)
    return L


def generate_synthetic_LFs(Y, m: int, seed: Optional[int] = None, good_ratio: float = 1 / 3,
                           p: Optional[float] = None, good_min_theta: float = 2, good_max_theta: float = 5,
                           bad_min_theta: float = 0.001, bad_max_theta: float = 0.01, sampler: str = "rim"):
    """A mix of accurate (``good_ratio``) and near-random Mallows LFs, thetas sorted ascending."""
    assert len(Y) != 0, "True label Y is empty"
    rng = np.random.default_rng(seed)
    m_good = int(m * good_ratio)
    thetas = np.sort(np.concatenate([rng.uniform(good_min_theta, good_max_theta, m_good),
                                     rng.uniform(bad_min_theta, bad_max_theta, m - m_good)]))
    return sample_mallows_LFs(Y, m, thetas, p, sampler=sampler, rng=rng)


def estimate_theta(L, train_method: str = "triplet_opt") -> np.ndarray:
    from .label_model import RankingLabelModel

    r_utils = L[0][0].r_utils
    return RankingLabelModel(r_utils, train_method=train_method).fit(L).thetas


def get_pair_wise_dists(L) -> list[np.ndarray]:
    """Per-example pairwise LF distance matrices (to reuse across inference calls)."""
    r_utils = L[0][0].r_utils
    return [r_utils.get_pair_wise_dists(row) for row in L]


def collect_outs(lst_out):
    out = defaultdict(list)
    for o in lst_out:
        for k in o:
            out["lst_" + k].append(o[k])
    return out


def mean_std(out, axis=0):
    o = {}
    for k in out:
        o[f"mean_{axis}_{k}"] = np.mean(out[k], axis=axis)
        o[f"std_{axis}_{k}"] = np.std(out[k], axis=axis)
    return o
