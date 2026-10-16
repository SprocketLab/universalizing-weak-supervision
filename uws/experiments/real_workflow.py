"""End-to-end ranking workflow on real datasets: LFs -> label model -> weak labels."""

from __future__ import annotations

import math
import os
import pickle
import warnings

import numpy as np

from ..label_model.ranking import RankingLabelModel
from ..labeling.ranking import FeatureRankingLF
from ..ranking.ranking import Ranking, RankingUtils
from ..synthetic.ranking import generate_synthetic_LFs

WEAK_LABEL_FILE_NAME = "weak_labels.pkl"
SYNTHETIC_WEAK_LABEL_FILE_NAME = "synthetic_weak_labels.pkl"


def generate_LFs(dataset, lst_labeling_functions):
    """Apply every LF to every item set of ``dataset`` (uses ``dataset.lst_ref_map``)."""
    return [[lf.apply(dataset.lst_ref_map[i]) for lf in lst_labeling_functions] for i in range(dataset.n)]


def snorkel_rankings(L, r_utils: RankingUtils, n_epochs: int = 500, seed: int = 0):
    """Snorkel baseline: every permutation is a class (only feasible for small ``d``)."""
    try:
        from snorkel.labeling.model import LabelModel
    except ImportError as e:  # pragma: no cover
        raise ImportError("The snorkel baseline needs `pip install universalizing-weak-supervision[snorkel]`") from e
    r_utils.set_perm2int_int2perm_mapping()
    L_int = np.array([r_utils.perm2int(row) for row in L])
    label_model = LabelModel(cardinality=math.factorial(r_utils.d), verbose=False)
    label_model.fit(L_train=L_int, n_epochs=n_epochs, log_freq=100, seed=seed)
    return r_utils.int2perm(label_model.predict(L_int, tie_break_policy="random"))


def get_weak_labels(dataset, weak_sup_conf: dict, root_path: str = "."):
    """Generate (or load) LF outputs and aggregate them. Returns ``(Y_hat, thetas, individual_kt)``.

    ``weak_sup_conf`` keys: ``lf_features``, ``lf_features_highest_first_flag``, ``num_LFs``,
    ``checkpoint_path``, ``recreate_if_exists``, ``synthetic`` (Mallows LFs instead of feature LFs, with
    optional ``good_*_theta``/``bad_*_theta``, ``p``), ``seed``, ``train_method``, ``inference_rule``
    (``snorkel`` for the baseline) and ``legacy`` (the paper code's Kendall distance, medoid "Kemeny",
    Mallows sampler and theta draws).
    """
    legacy = bool(weak_sup_conf.get("legacy", False))
    m = weak_sup_conf.get("num_LFs") or len(weak_sup_conf["lf_features"])
    seed = weak_sup_conf.get("seed")
    if seed is None:
        warnings.warn("No random seed in weak_sup_conf; results are not reproducible", stacklevel=2)
    d = dataset.d
    lst_lfs = [FeatureRankingLF(f, d, flag) for f, flag in
               zip(weak_sup_conf["lf_features"], weak_sup_conf["lf_features_highest_first_flag"])]

    root_path = dataset.data_conf.get("project_root", root_path)
    weak_label_path = os.path.join(root_path, weak_sup_conf["checkpoint_path"])
    synthetic = weak_sup_conf.get("synthetic") is True
    weak_label_file = os.path.join(weak_label_path,
                                   SYNTHETIC_WEAK_LABEL_FILE_NAME if synthetic else WEAK_LABEL_FILE_NAME)

    if weak_sup_conf["recreate_if_exists"] or not os.path.exists(weak_label_file):
        os.makedirs(weak_label_path, exist_ok=True)
        if synthetic:
            kw = {k: weak_sup_conf[k] for k in ("good_max_theta", "good_min_theta", "bad_max_theta",
                                                 "bad_min_theta") if weak_sup_conf.get(k) is not None}
            L = generate_synthetic_LFs(dataset.Y, m, seed=seed, p=weak_sup_conf.get("p"), legacy_thetas=legacy,
                                       sampler="legacy" if legacy else "rim", **kw)
        else:
            L = generate_LFs(dataset, lst_lfs)
        with open(weak_label_file, "wb") as fd:
            pickle.dump(L, fd)
        print("Weak labels generated and saved in", weak_label_file)
    else:
        print("Weak labels found in", weak_label_file, "Load it...")
        with open(weak_label_file, "rb") as fd:
            L = pickle.load(fd)

    r_utils = RankingUtils(d, legacy=legacy)
    m = len(L[0])
    individual_kt = [[r_utils.mean_kt_distance([row[j] for row in L], dataset.Y) for j in range(m)]]

    if weak_sup_conf.get("inference_rule", "").lower() == "snorkel":
        return snorkel_rankings(L, r_utils), [], individual_kt

    print(f"UWS label model: train_method={weak_sup_conf['train_method']}, "
          f"inference_rule={weak_sup_conf['inference_rule']}")
    lm = RankingLabelModel(r_utils, weak_sup_conf["train_method"], weak_sup_conf["inference_rule"], seed=seed)
    lm.fit(L)
    return lm.predict(L), lm.thetas, individual_kt


def restore_ranking(scores):
    """Scores (n x d, lower = better) -> rankings."""
    assert len(scores) > 0
    d = len(scores[0])
    r_utils = RankingUtils(d)
    return [Ranking([i for i, _ in sorted(enumerate(y), key=lambda x: x[1])], r_utils) for y in scores]


def convert_to_torch(X, Y_true, lst_pi_hat):
    import torch

    return (torch.tensor(X).float(),
            torch.tensor([y.permutation for y in Y_true], dtype=torch.float32),
            torch.tensor([y.permutation for y in lst_pi_hat], dtype=torch.float32))


def shuffle_ranking(X, Y_true, L, generator=None):
    """Shuffle item order per example (otherwise the true label is always 0, 1, ..., d-1)."""
    import torch

    n, d = X.shape[0], X.shape[1]
    for i in range(n):
        perm = torch.randperm(d, generator=generator)
        X[i], Y_true[i], L[i] = X[i][perm], Y_true[i][perm], L[i][perm]
    return X, Y_true, L


def get_mean_kt_distance(pred, ground_truth):
    d = pred.shape[1]
    return RankingUtils(d).mean_kt_distance(restore_ranking(pred), restore_ranking(ground_truth))
