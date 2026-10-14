"""Synthetic classification-as-ranking experiment: rank classes by distance to each class cluster, then
compare the ranking label model (Mallows LFs) with Snorkel on classification LFs."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from ..ranking.label_model import RankingLabelModel
from ..ranking.mallows import Mallows
from ..ranking.ranking import Ranking, RankingUtils


def generate_dataset(num_pts: int, num_classes: int, seed: int = 1):
    X, y = make_classification(n_samples=num_pts, n_classes=num_classes, n_informative=10, random_state=seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, random_state=seed)
    return X_train, Y_train, X_test, Y_test


def transformToRankedLabels(X, Y_cls):
    """Rank classes for each point by its distance to the nearest member of each class."""
    clusters = defaultdict(list)
    for x, c in zip(X, Y_cls):
        clusters[c].append(x)
    classes = sorted(clusters)
    r_utils = RankingUtils(len(classes))
    centers = {c: np.stack(clusters[c]) for c in classes}
    Y_ranks = []
    for x in X:
        dists = [(c, np.linalg.norm(centers[c] - x, axis=1).min()) for c in classes]
        Y_ranks.append(Ranking([c for c, _ in sorted(dists, key=lambda t: t[1])], r_utils))
    return Y_ranks


def generate_ranking_LFs(Y_ranks, thetas, rng=None):
    r_utils = Y_ranks[0].r_utils
    lst_mlw = [Mallows(r_utils, theta, rng=rng) for theta in thetas]
    return [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y_ranks]


def generate_class_LFs(lst_P, Y_cls, rng=None):
    """``P[i][j]``: probability of outputting label ``i - 1`` given class ``j`` (row 0 = abstain)."""
    rng = np.random.default_rng() if rng is None else rng
    return np.array([[int(np.nonzero(rng.multinomial(1, P[:, y]))[0][0]) - 1 for P in lst_P] for y in Y_cls])


def get_LF_class_probs(lst_acc, lst_abstains, num_classes):
    lst_P = []
    for acc, ab in zip(lst_acc, lst_abstains):
        P = np.full((num_classes, num_classes), (1 - acc) / (num_classes - 1))
        np.fill_diagonal(P, acc)
        P = P * (1 - ab)
        lst_P.append(np.vstack((np.full(num_classes, ab), P)))
    return lst_P


def evaluate(Y_tilde_ranks, Y_class, lst_k=(1,)):
    """Top-k accuracy of the class rankings (score of the item at rank r is 1/(r+1))."""
    d = len(Y_tilde_ranks[0])
    scores = np.zeros((len(Y_tilde_ranks), d))
    for i, y in enumerate(Y_tilde_ranks):
        for r, c in enumerate(list(y)):
            scores[i, c] = 1.0 / (r + 1)
    return [top_k_accuracy_score(Y_class, scores, k=k, labels=np.arange(d)) for k in lst_k]


def run_ws_workflow(X_train, Y_train, conf, seed: int = 0):
    """``conf['ws_method']``: ``rankings`` (needs ``thetas``) or ``snorkel`` (``lst_acc``, ``lst_abstains``)."""
    rng = np.random.default_rng(seed)
    d = conf["num_classes"]
    out = {}
    Y_train_ranks = transformToRankedLabels(X_train, Y_train)
    r_utils = Y_train_ranks[0].r_utils

    if conf["ws_method"] == "rankings":
        thetas = np.asarray(conf["thetas"])
        L = generate_ranking_LFs(Y_train_ranks, thetas, rng)
        lm = RankingLabelModel(r_utils, conf.get("train_method", "triplet_opt"),
                               conf.get("inference_rule", "weighted_kemeny"), seed=seed).fit(L)
        Y_train_ranks_tilde = lm.predict(L)
        out["parameter_error"] = float(np.linalg.norm(lm.thetas - thetas))
    elif conf["ws_method"] == "snorkel":
        from snorkel.labeling.model import LabelModel

        lst_P = get_LF_class_probs(conf["lst_acc"], conf["lst_abstains"], d)
        L = generate_class_LFs(lst_P, Y_train, rng)
        label_model = LabelModel(cardinality=d, verbose=False)
        label_model.fit(L_train=L, n_epochs=500, log_freq=100, seed=123)
        _, probs = label_model.predict(L, return_probs=True)
        Y_train_ranks_tilde = [Ranking(np.argsort(-p), r_utils) for p in probs]
        est = label_model.get_conditional_probs()[:, 1:, :].reshape(-1, d)
        true = np.vstack([P[1:] for P in lst_P])
        out["parameter_error"] = float(np.linalg.norm(est - true))
    else:
        raise ValueError(f"Unknown ws_method {conf['ws_method']!r}")

    Y_train_class_tilde = [y[0] for y in Y_train_ranks_tilde]
    out["mean_kt"] = r_utils.mean_kt_distance(Y_train_ranks, Y_train_ranks_tilde)
    for k, acc in zip((1, 2, 3), evaluate(Y_train_ranks_tilde, Y_train, lst_k=[1, 2, 3])):
        out[f"ac-{k}"] = acc
    out["per_class_acc"] = confusion_matrix(Y_train, Y_train_class_tilde, normalize="true").diagonal()
    counts = np.bincount(Y_train)
    out["class_count"] = [(i, counts[i]) for i in np.nonzero(counts)[0]]
    return Y_train_class_tilde, out


def estimate_theta_acc(theta, d, n_samples: int = 4000, rng=None):
    """Monte-Carlo probability that a Mallows(theta) sample keeps the center's item at ranks 0, 1, 2."""
    r_utils = RankingUtils(d)
    rng = np.random.default_rng() if rng is None else rng
    mlw = Mallows(r_utils, theta, rng=rng)
    y = r_utils.get_random_ranking(rng)
    S = np.array([mlw.sample(y)[0].permutation for _ in range(n_samples)])
    return tuple(float(np.mean(S[:, k] == y[0])) for k in range(3))


def build_dict_d_theta_mu(lst_d, lst_thetas, save_to_file_name):
    """Cache of ``estimate_theta_acc`` keyed by ``d`` then ``int(100 theta)``, updated in place on disk."""
    cache = {}
    if os.path.isfile(save_to_file_name):
        with open(save_to_file_name, "rb") as f:
            cache = pickle.load(f)
    for d in lst_d:
        per_theta = cache.setdefault(d, {})
        for theta in lst_thetas:
            key = int(theta * 100)
            if key not in per_theta:
                print("estimating for", d, theta)
                per_theta[key] = estimate_theta_acc(theta, d)
    with open(save_to_file_name, "wb") as f:
        pickle.dump(cache, f)
    return cache


def load_dict_d_theta_mu(load_from_file_name):
    with open(load_from_file_name, "rb") as f:
        return pickle.load(f)
