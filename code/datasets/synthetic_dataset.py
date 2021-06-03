import sys

sys.path.append('../')
import pandas as pd
import numpy as np
import torch
import copy
from sklearn.model_selection import train_test_split
from basic_clmn_dataset import *
from ranking_utils import *


class SyntheticRankingDataset:
    def __init__(self, data_conf):

        raw_file_path = '{}/{}'.format(data_conf['project_root'], data_conf['raw_data_path'])
        self.base_dataset = BasicColumnarDataset(raw_file_path, id_feature='id',
                                                 features_subset=data_conf['features'])
        self.base_dataset.preprocess()  # might need data_conf params

        self.n = data_conf['num_samples']
        self.d = data_conf['dimension']
        self.lst_id = self.base_dataset.get_all_key_values()
        self.base_dataset.label_feature = data_conf['label_feature']
        self.highest_first = data_conf['highest_first']

        self.data_conf = data_conf
        self.seed = data_conf['seed']
        self.train_fraction = data_conf['train_fraction']
        self.r_utils = RankingUtils(self.d)

    def create_samples(self):
        """

        Returns
        -------

        """
        print("Started create_samples")
        self.Y_true = []
        self.lst_id_map = []
        self.lst_feature_map = []
        self.X = []
        bd = self.base_dataset

        np.random.seed(self.seed)

        for i in range(self.n):
            sel_idcs = np.random.choice(self.lst_id, self.d, replace=False)
            f_map = [bd.get_feature_map(idx) for idx in sel_idcs]
            id_map = dict(zip(sel_idcs, range(self.d)))

            y_ = [ ( id_map[idx], bd.get_label(idx)) for idx in sel_idcs]
            y_ = sorted(y_, key=lambda x: x[1], reverse=self.highest_first)
            y  = [i for i,j in y_]

            self.Y_true.append(Ranking(y, self.r_utils))
            self.lst_id_map.append(id_map)
            self.lst_feature_map.append(f_map)
        print("Finished create_samples")
        self.feature_map_to_np_array()

    def feature_map_to_np_array(self):
        fs = sorted(self.lst_feature_map[0][0].keys())
        X = [[[self.lst_feature_map[i][j][k] for k in fs] for j in range(self.d)] for i in range(self.n)]
        self.X = np.array(X)

    def set_Y_tilde(self, Y_tilde):
        """
        Set Y_tilde which comes from weak superivison estimation
        Parameters
        ----------
        Y_tilde

        Returns
        -------

        """
        self.Y_tilde = Y_tilde

    def get_train_test_torch(self, use_weak_labels):
        """
        return X_train, X_test, Y_train, Y_test
        if use_weak_labels is True, Y_train uses Y_tilde instead of Y_true (test_Y still comes from Y_true)
        Parameters
        ----------
        use_weak_labels: l2r training conf

        Returns
        -------

        """
        if not hasattr(self, "X"):
            AssertionError("create_samples should run first")

        if (use_weak_labels) and (not hasattr(self, "Y_tilde")):
            AssertionError("Y_tilde should be set before using it through set_Y_tilde")


        X_torch = torch.tensor(self.X)

        Y_true_ranking = self.perm2ranking(self.Y_true)

        Y_true_torch = torch.tensor(Y_true_ranking, dtype=float).float()

        # ranking to score
        Y_true_torch = self._torch_ranking_to_score(Y_true_torch)

        if use_weak_labels:
            Y_tilde_ranking = self.perm2ranking(self.Y_tilde)
            Y_tilde_torch = torch.tensor(Y_tilde_ranking, dtype=float).float()
            # ranking to score
            Y_tilde_torch = self._torch_ranking_to_score(Y_tilde_torch)

        X_train, X_test, Y_train, Y_test = train_test_split(X_torch, Y_true_torch, train_size=self.train_fraction,
                                                            random_state=self.seed)
        if use_weak_labels:
            _, _, Y_train, _ = train_test_split(X_torch, Y_tilde_torch, train_size=self.train_fraction,
                                                random_state=self.seed)
            print(f"use_weak_labels:{use_weak_labels}, we will use weak labels")
        else:
            print(f"use_weak_labels:{use_weak_labels}, we will use true labels")
        return X_train, X_test, Y_train, Y_test

    def _torch_ranking_to_score(self, Y):
        """

        Returns
        -------

        """
        Y = self.d - Y  # reverse the ranking for scoring
        return Y

    def perm2ranking(self, Y):
        """
        Convert permutation ranking
        In permutation, (2, 0, 1) means top ranking is placed at feature index 2, the second ranking is placed
        at feature index 0, so on.
        In ranking, (2, 0, 1) means first feature index has the ranking 2,
        the second index has the ranking 0 (top ranking), so on.
        By inverting them, make a proper label for learning to rank

        Parameters
        ----------
        Y

        Returns
        -------

        """
        # permutation to ranking
        Y_perm = [y.permutation for y in Y]

        # make invert dictionary
        Y_ranking = []
        for i in range(self.n):
            invert_pairs = zip(Y_perm[i], range(self.d)) # the dict of index -> ranking

            # sort based on index
            sorted_invert_pairs = sorted(invert_pairs, key=lambda x: x[0])
            y_ranking = [ranking for idx, ranking in sorted_invert_pairs]
            Y_ranking.append(y_ranking)

        return Y_ranking