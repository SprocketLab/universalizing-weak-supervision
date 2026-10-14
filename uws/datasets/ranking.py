"""Ranking datasets: random sets of ``d`` items, ranked by a label column (board games, movies)."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.stats import percentileofscore
from sklearn.model_selection import train_test_split

from ..ranking.ranking import Ranking, RankingUtils, ranking_to_score
from .columnar import BasicColumnarDataset

_SAMPLE_KEYS = ("lst_id_train", "lst_id_test", "lst_feature_map_train", "lst_feature_map_test",
                "lst_ref_map_train", "lst_ref_map_test", "X_train", "Y_train", "X_test", "Y_test")


class ColumnarRankingDataset:
    """Samples ``n_train + n_test`` item sets of size ``dimension`` from a CSV and ranks them by
    ``label_feature``. Train and test sets draw from disjoint item pools.

    ``data_conf`` keys: ``project_root``, ``raw_data_path``, ``processed_data_path``, ``features``,
    ``label_feature``, ``highest_first``, ``n_train``, ``n_test``, ``dimension``, ``seed``,
    ``recreate_if_exists`` and optionally ``tiered`` (one item per label quantile).
    """

    id_feature = "id"

    def __init__(self, data_conf: dict):
        raw_file_path = os.path.join(data_conf["project_root"], data_conf["raw_data_path"])
        self.base_dataset = BasicColumnarDataset(raw_file_path, id_feature=self.id_feature,
                                                 features_subset=data_conf["features"])
        self.base_dataset.preprocess()
        self.n_train = data_conf["n_train"]
        self.n_test = data_conf["n_test"]
        self.n = self.n_train + self.n_test
        self.d = data_conf["dimension"]
        self.lst_id = self.base_dataset.get_all_key_values()
        self.base_dataset.label_feature = data_conf["label_feature"]
        self.highest_first = data_conf["highest_first"]
        self.tiered = data_conf.get("tiered", False)
        self.data_conf = data_conf
        self.processed_data_path = os.path.join(data_conf["project_root"], data_conf["processed_data_path"])
        self.seed = data_conf["seed"]
        self.r_utils = RankingUtils(self.d)

    def create_samples(self):
        """Create (or load cached) samples; sets ``X_train/Y_train/X_test/Y_test`` and concatenations."""
        sample_path = os.path.join(self.processed_data_path, "sample.pkl")
        if os.path.exists(sample_path) and not self.data_conf["recreate_if_exists"]:
            print(f"Saved samples found in {sample_path} and recreate_if_exists=False, load the data...")
            with open(sample_path, "rb") as f:
                dump = pickle.load(f)
            # older caches (movies) used lst_movie_id_*
            for split in ("train", "test"):
                if f"lst_movie_id_{split}" in dump:
                    dump[f"lst_id_{split}"] = dump.pop(f"lst_movie_id_{split}")
            for k in _SAMPLE_KEYS:
                setattr(self, k, dump[k])
        else:
            print("Generate samples...")
            self.lst_id_train, self.lst_id_test = train_test_split(
                self.lst_id, train_size=self.n_train / self.n, random_state=self.seed)
            rng = np.random.default_rng(self.seed)
            (self.X_train, self.Y_train, self.lst_feature_map_train,
             self.lst_ref_map_train) = self._create_samples(self.lst_id_train, self.n_train, rng)
            (self.X_test, self.Y_test, self.lst_feature_map_test,
             self.lst_ref_map_test) = self._create_samples(self.lst_id_test, self.n_test, rng)
            os.makedirs(self.processed_data_path, exist_ok=True)
            with open(sample_path, "wb") as f:
                pickle.dump({k: getattr(self, k) for k in _SAMPLE_KEYS}, f)

        self.X = np.vstack((self.X_train, self.X_test))
        self.Y = self.Y_train + self.Y_test
        self.lst_feature_map = self.lst_feature_map_train + self.lst_feature_map_test
        self.lst_ref_map = self.lst_ref_map_train + self.lst_ref_map_test

    def _create_samples(self, ids, n: int, rng: np.random.Generator):
        bd = self.base_dataset
        ids = np.asarray(ids)
        if self.tiered:
            labels = np.array([bd.get_label(i) for i in ids])
            pct = np.array([percentileofscore(labels, x) for x in labels])
            unit = 100 / self.d
            levels = [ids[(pct >= unit * k) & (pct < unit * (k + 1) + (k == self.d - 1))] for k in range(self.d)]
        Y, feats, refs = [], [], []
        for _ in range(n):
            if self.tiered:
                sel = rng.permutation([rng.choice(level) for level in levels])
            else:
                sel = rng.choice(ids, self.d, replace=False)
            id_map = {idx: k for k, idx in enumerate(sel)}
            y = sorted(((id_map[idx], bd.get_label(idx)) for idx in sel), key=lambda x: x[1],
                       reverse=self.highest_first)
            Y.append(Ranking([k for k, _ in y], self.r_utils))
            feats.append([bd.get_feature_map(idx) for idx in sel])
            refs.append([bd.get_ref_map(idx) for idx in sel])
        return self._feature_map_to_np_array(feats), Y, feats, refs

    def _feature_map_to_np_array(self, lst_feature_map):
        fs = sorted(lst_feature_map[0][0].keys())
        return np.array([[[fm[k] for k in fs] for fm in row] for row in lst_feature_map], dtype=float)

    def set_Y_tilde(self, Y_tilde):
        """Set the weak labels (aggregated LF rankings) used by ``get_train_test_torch``."""
        self.Y_tilde = Y_tilde

    def get_train_test_torch(self, use_weak_labels: bool):
        """``X_train, X_test, Y_train, Y_test`` as torch tensors (scores); Y_train from weak labels if asked."""
        import torch

        if not hasattr(self, "X"):
            raise RuntimeError("create_samples() must run first")
        if use_weak_labels and not hasattr(self, "Y_tilde"):
            raise RuntimeError("set_Y_tilde() must be called before using weak labels")
        X_train, X_test = torch.tensor(self.X_train), torch.tensor(self.X_test)
        Y_src = self.Y_tilde[: self.n_train] if use_weak_labels else self.Y_train
        return X_train, X_test, ranking_to_score(Y_src, self.d, False), ranking_to_score(self.Y_test, self.d, False)


class BoardGamesRankingDataset(ColumnarRankingDataset):
    id_feature = "id"


class MoviesRankingDataset(ColumnarRankingDataset):
    id_feature = "imdbId"


class SyntheticRankingDataset(ColumnarRankingDataset):
    """Like the real datasets (used with synthetic Mallows LFs); ``num_samples`` and ``train_fraction``
    may replace ``n_train``/``n_test``."""

    def __init__(self, data_conf: dict):
        conf = dict(data_conf)
        if "num_samples" in conf and "n_train" not in conf:
            n_train = int(conf["num_samples"] * conf.get("train_fraction", 0.5))
            conf["n_train"], conf["n_test"] = n_train, conf["num_samples"] - n_train
        conf.setdefault("processed_data_path", "data/synthetic/processed")
        conf.setdefault("recreate_if_exists", True)
        super().__init__(conf)


def create_dataset(data_conf: dict):
    name = data_conf["dataset_name"]
    classes = {"movies": MoviesRankingDataset, "boardgames": BoardGamesRankingDataset,
               "synthetic": SyntheticRankingDataset}
    if name not in classes:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(classes)}")
    return classes[name](data_conf)
