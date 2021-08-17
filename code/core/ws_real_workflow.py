import sys
sys.path.append('../')
from ranking_utils import RankingUtils
from mallows import Mallows
from ws_ranking import WeakSupRanking
from synth_ranking_utils import sample_mallows_LFs
from labelling.feature_lf import FeatureRankingLF
from snorkel.labeling.model import LabelModel
import os
import numpy as np
import pickle
import torch
import math


WEAK_LABEL_FILE_NAME  = 'weak_labels.pkl'
SYNTHETIC_WEAK_LABEL_FILE_NAME = 'synthetic_weak_labels.pkl'

def generate_synthetic_LFs(Y, m, p=None, min_theta=0, max_theta=1.0):
    """

    Parameters
    ----------
    Y: true label
    m: the number of label functions
    p: the fraction of unmasked items
    min_theta: the min value of uniform sampling for thetas
    max_theta: the max value of uniform sampling for thetas

    Returns
    -------
    """
    assert len(Y) != 0, "True label Y is empty"
    np.random.seed(0)
    thetas = np.random.uniform(min_theta, max_theta, m)
    thetas = np.sort(thetas)

    L = sample_mallows_LFs(Y, m, thetas, p)
    return L

def generate_LFs(dataset, lst_labeling_functions):
    """

    Parameters
    ----------
    dataset
    lst_labeling_functions

    Returns
    -------

    """
    L = []
    n = dataset.n
    for i in range(n):
        l = []
        for lf in lst_labeling_functions:
            l.append(lf.apply(dataset.lst_ref_map[i]))
        L.append(l)
    
    return L

def get_weak_labels(dataset, weak_sup_conf, root_path='.'):
    """
    get Y weak labels based on lf_features (the list of feature) - 
    It just generarte Rankings based on features and aggregates according to params in
    weak_sup_conf
    Parameters
    ----------
    dataset
    weak_sup_conf

    Returns
    -------

    """
    # infer n and d
    
    n = dataset.n
    if weak_sup_conf.get('num_LFs') is None:
        m = len(weak_sup_conf['lf_features'])
    else:
        m = weak_sup_conf['num_LFs']

    d = dataset.d
    lf_features = weak_sup_conf['lf_features']
    lf_features_flags = weak_sup_conf['lf_features_highest_first_flag']

    lst_lfs = [FeatureRankingLF(feature, d, highest_first_flag)
                        for feature,highest_first_flag in zip(lf_features,lf_features_flags)]

    root_path = dataset.data_conf['project_root']
    weak_label_path = os.path.join(root_path, weak_sup_conf['checkpoint_path'])

    # generate weak labels - if checkpoint exists, load it.
    if weak_sup_conf.get('synthetic') is True:
        weak_label_file_path = os.path.join(weak_label_path, SYNTHETIC_WEAK_LABEL_FILE_NAME)
    else:
        weak_label_file_path = os.path.join(weak_label_path, WEAK_LABEL_FILE_NAME)

    if (weak_sup_conf['recreate_if_exists']) or (not os.path.exists(weak_label_file_path)):
        if not os.path.exists(weak_label_path):
            os.makedirs(weak_label_path, exist_ok=True)

        # generate weak labels
        if weak_sup_conf.get('synthetic') is True:
            L = generate_synthetic_LFs(dataset.Y, m, p=weak_sup_conf.get('p'))
        else:
            L = generate_LFs(dataset, lst_lfs)

        # dump generated weak labels
        with open(weak_label_file_path, 'wb') as fd:
            pickle.dump(L, fd)
        print("Weak labels generated and saved in", weak_label_file_path)
    else:
        # load weak labels
        with open(weak_label_file_path, 'rb') as fd:
            print("Weak labels found in", weak_label_file_path, "Load it...")
            L = pickle.load(fd)

    r_utils = RankingUtils(d)

    if weak_sup_conf.get('inference_rule').lower() == 'snorkel':
        print("Use snorkel...")
        r_utils.set_perm2int_int2perm_mapping()
        L_int = np.array([r_utils.perm2int(lstRanks) for lstRanks in L])
        label_model = LabelModel(cardinality=math.factorial(d), verbose=True)
        label_model.fit(L_train=L_int, n_epochs=500, log_freq=100)
        lst_pi_hat = r_utils.int2perm(label_model.predict(L_int, tie_break_policy="random"))
        return lst_pi_hat, []
    else:
        print(f"Use our weak supervision...train_method: {weak_sup_conf['train_method']},"
              f"inference_rule: {weak_sup_conf['inference_rule']}")
        wsr = WeakSupRanking(r_utils)
        wsr.train(weak_sup_conf, L)

        m = len(L[0])
        lst_pi_hat = wsr.infer_ranking(weak_sup_conf, L, numLFs=m)

        return lst_pi_hat, wsr.thetas


def restore_ranking(scores):
    """
    Restore ranking from scores
    will be deprecated because it is not necessary when we are using ptranking
    Parameters
    ----------
    scores

    Returns
    -------

    """
    # infer d from scores
    assert len(scores) > 0
    d = len(scores[0])

    r_utils = RankingUtils(d)
    ranking = []
    for y in scores:
        y_ = [(i, y[i]) for i in range(len(y))]
        y_ = list(sorted(y_, key=lambda x: x[1]))
        ranking.append(Ranking([i[0] for i in y_], r_utils))
    return ranking


def convert_to_torch(X, Y_true, lst_pi_hat):
    """
    Convert X, Y_true, estimated Y into torch tensor

    Parameters
    ----------
    X
    Y_true
    lst_pi_hat

    Returns
    -------

    """
    X = torch.tensor(X).float()
    Y_true = torch.tensor([y.permutation for y in Y_true], dtype=float).float()
    L = torch.tensor([y.permutation for y in lst_pi_hat], dtype=float).float()
    return X, Y_true, L


def shuffle_ranking(X, Y_true, L):
    """
    Shuffle labels if necessary (without this, Y_true label fixed as 0, 1, 2, ..., d-1
    Parameters
    ----------
    X
    Y_true
    L

    Returns
    -------

    """
    # infer n and d
    n = X.shape[0]
    d = X.shape[1]

    for i in range(n):
        perm = torch.randperm(d)
        X[i] = X[i][perm]
        Y_true[i] = Y_true[i][perm]
        L[i] = L[i][perm]
    return X, Y_true, L


def get_mean_kt_distance(pred, ground_truth):
    """
    get mean kendall tau distance
    Parameters
    ----------
    pred
    ground_truth

    Returns
    -------

    """
    d = pred.shape[1]
    r_utils = RankingUtils(d)
    pred_ranking = restore_ranking(pred)
    ground_truth_ranking = restore_ranking(ground_truth)

    return r_utils.mean_kt_distance(pred_ranking, ground_truth_ranking)