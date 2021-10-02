import sys
import os

sys.path.append('../')
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from basic_clmn_dataset import *
from ranking_utils import *


class BoardGamesRankingDataset:
    def __init__(self, data_conf):
        raw_file_path = '{}/{}'.format(data_conf['project_root'], data_conf['raw_data_path'])
        self.base_dataset = BasicColumnarDataset(raw_file_path, id_feature='id',
                                                 features_subset=data_conf['features'])

        self.base_dataset.preprocess()  # might need data_conf params
        self.n_train = data_conf['n_train']
        self.n_test = data_conf['n_test']
        self.n = self.n_train + self.n_test
        self.d = data_conf['dimension']
        self.lst_id = self.base_dataset.get_all_key_values()
        self.base_dataset.label_feature = data_conf['label_feature']
        self.highest_first = data_conf['highest_first']

        self.data_conf = data_conf
        self.processed_data_path = os.path.join(data_conf['project_root'], data_conf['processed_data_path'])
        self.seed = data_conf['seed']
        self.r_utils = RankingUtils(self.d)

    def create_samples(self):
        """
        Create samples
        As a result, the following attributes are set

        self.lst_id_train
        self.lst_id_test
        self.lst_feature_map_train,
        self.lst_feature_map_test,
        self.X_train
        self.Y_train
        self.X_test
        self.Y_test

        Returns
        -------

        """
        sample_path = os.path.join(self.processed_data_path, 'sample.pkl')

        # data load (file exists & recreat_if_exists = False)
        if os.path.exists(sample_path) and (not self.data_conf['recreate_if_exists']):
            print(f'Saved samples found in {sample_path} and recreate_if_exists=False, load the data...')
            with open(sample_path, 'rb') as f:
                dict_pickle = pickle.load(f)
            self._set_sample_pickle(dict_pickle)
        # sample generation
        else:
            print('Generate samples...')
            # separate train, test id
            train_fraction = self.n_train / self.n
            self.lst_id_train, self.lst_id_test = train_test_split(self.lst_id,
                                                                   train_size=train_fraction,
                                                                   random_state=self.seed)
            self.X_train, self.Y_train, self.lst_feature_map_train, self.lst_ref_map_train = self._create_samples(self.lst_id_train,
                                                                                          train=True)
            self.X_test, self.Y_test, self.lst_feature_map_test, self.lst_ref_map_test = self._create_samples(self.lst_id_test,
                                                                                       train=False)
            dict_to_dump = {'lst_id_train': self.lst_id_train,
                            'lst_id_test': self.lst_id_test,
                            'lst_feature_map_train': self.lst_feature_map_train,
                            'lst_feature_map_test': self.lst_feature_map_test,
                            'lst_ref_map_train': self.lst_ref_map_train,
                            'lst_ref_map_test': self.lst_ref_map_test,
                            'X_train': self.X_train,
                            'Y_train': self.Y_train,
                            'X_test': self.X_test,
                            'Y_test': self.Y_test}
            # save generated samples
            if not os.path.exists(self.processed_data_path):
                os.makedirs(self.processed_data_path)
            with open(sample_path, 'wb') as f:
                pickle.dump(dict_to_dump, f)

        # concatenate data so that we can infer aggregated weak label from the whole dataset
        self.X = np.vstack((self.X_train, self.X_test))
        self.Y = self.Y_train + self.Y_test
        self.lst_feature_map = self.lst_feature_map_train + self.lst_feature_map_test
        self.lst_ref_map = self.lst_ref_map_train + self.lst_ref_map_test

    def _create_samples(self, lst_id_sets, train):
        """
        inner logic of create_samples

        Parameters
        ----------
        lst_id_sets
        train

        Returns
        -------

        """
        Y = []
        lst_id_map = []
        lst_feature_map = []
        lst_ref_map = []

        bd = self.base_dataset

        np.random.seed(self.seed)

        # set the sample number
        if train:
            n = self.n_train
        else:
            n = self.n_test

        # generate features & labels of item sets
        for i in range(n):
            sel_idcs = np.random.choice(lst_id_sets, self.d, replace=False)
            id_map = dict(zip(sel_idcs, range(self.d)))
            f_map = [bd.get_feature_map(idx) for idx in sel_idcs]
            ref_map = [bd.get_ref_map(idx) for idx in sel_idcs]
            

            y_ = [(id_map[idx], bd.get_label(idx)) for idx in sel_idcs]
            y_ = sorted(y_, key=lambda x: x[1], reverse=self.highest_first)
            y = [i for i, j in y_]

            Y.append(Ranking(y, self.r_utils))
            lst_id_map.append(id_map)
            lst_feature_map.append(f_map)
            lst_ref_map.append(ref_map)

        X = self._feature_map_to_np_array(lst_feature_map, n)
        return X, Y, lst_feature_map, lst_ref_map

    def _set_sample_pickle(self, dict_pickle):
        """
        set sample attributes from loaded data
        Parameters
        ----------
        dict_pickle

        Returns
        -------

        """
        self.lst_movie_id_train = dict_pickle['lst_movie_id_train']
        self.lst_movie_id_test = dict_pickle['lst_movie_id_test']
        self.lst_feature_map_train = dict_pickle['lst_feature_map_train']
        self.lst_feature_map_test = dict_pickle['lst_feature_map_test']
        self.lst_ref_map_train = dict_pickle['lst_ref_map_train']
        self.lst_ref_map_test = dict_pickle['lst_ref_map_test']
        self.X_train = dict_pickle['X_train']
        self.Y_train = dict_pickle['Y_train']
        self.X_test = dict_pickle['X_test']
        self.Y_test = dict_pickle['Y_test']

    def _feature_map_to_np_array(self, lst_feature_map, n):
        """
        Convert feature map to numpy array
        Parameters
        ----------
        lst_feature_map
        n

        Returns
        -------

        """
        fs = sorted(lst_feature_map[0][0].keys())
        X = [[[lst_feature_map[i][j][k] for k in fs] for j in range(self.d)] for i in range(n)]
        X = np.array(X)
        return X

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

        X_train = torch.tensor(self.X_train)
        X_test = torch.tensor(self.X_test)

        if use_weak_labels:  # use weak label for training
            print(f"use_weak_labels:{use_weak_labels}, we will use weak labels")
            Y_train = self._ranking_to_score(self.Y_tilde[:self.n_train])
        else:
            print(f"use_weak_labels:{use_weak_labels}, we will use true labels")
            Y_train = self._ranking_to_score(self.Y_train)
        Y_test = self._ranking_to_score(self.Y_test)

        return X_train, X_test, Y_train, Y_test

    def _ranking_to_score(self, Y, highest_first=False):
        """

        Returns
        -------

        """
        d = self.d
        Y_score_torch = ranking_to_score(Y, d, highest_first)
        return Y_score_torch