import sys
sys.path.append('../') 
#sys.path.append('../snorkel/') 
from ranking_utils import *
from mallows import *
from ws_ranking import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from snorkel.labeling.model import LabelModel
from collections import defaultdict

#from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import os
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def generate_dataset(num_pts, num_classes):
    """

    Parameters
    ----------
    num_pts: the number of samples
    num_classes: the number of classes

    Returns
    -------

    """
    # X dim (num_pts, n_informative (10))
    # y dim (num_pts,)
    X, y = make_classification(n_samples=num_pts, n_classes=num_classes, n_informative=10,random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y,random_state=1)
    return X_train, Y_train, X_test, Y_test

# obtain ranking using nearest neighbor
def transformToRankedLabels(X, Y_cls):
    """
    transform Y_cls to Y_ranks based on distances to each cluster
    Parameters
    ----------
    X
    Y_cls

    Returns
    -------

    """
    n = len(Y_cls)

    # collect clusters
    clusters = defaultdict(list)
    for i in range(n):
        clusters[Y_cls[i]].append(X[i])
    classes = list(sorted(clusters.keys()))
    logger.debug(classes)

    # determine ranks based on the distances to each cluster
    Y_ranks = []
    for i in range(n):
        # calculate distances to the clusters (take min distance)
        dists = [(cls, min([np.linalg.norm(X[i]-clusters[cls][j])
                            for j in range(len(clusters[cls]))])) for cls in classes]
        dists = sorted(dists, key=lambda x: x[1])
        ranks = Ranking([x[0] for x in dists])
        Y_ranks.append(ranks)

    return Y_ranks

def generate_ranking_LFs(Y_ranks, thetas):
    """
    Generate ranking LFs from Y_ranks and parameter thetas
    Parameters
    ----------
    Y_ranks
    thetas

    Returns
    -------

    """
    d = len(Y_ranks[0])
    r_utils = RankingUtils(d)

    lst_mlw = [Mallows(r_utils,theta) for theta in thetas ]
    L = [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y_ranks]

    return L

def generate_class_LFs(lst_P, Y_cls):
    """
    lst_P has a P matrix for each LF
    P[i][j] = prob of outputing label i given the true class is j ; i>0
    P[i][j] = prob of outputing Abstain  , i = 0
    Parameters
    ----------
    lst_P
    Y_cls

    Returns
    -------

    """

    L = []
    
    for y in Y_cls:
        lfs = []
        for P in lst_P:
            y_ = np.nonzero(np.random.multinomial(1, P[:,y]))[0][0] - 1
            lfs.append(y_)
        L.append(lfs)
    L = np.array(L)
    return L


def get_LF_class_probs(lst_acc,lst_abstains,num_classes):
    """

    Parameters
    ----------
    lst_acc
    lst_abstains
    num_classes

    Returns
    -------

    """
    m = len(lst_acc)
    card = num_classes
    lst_P = []
    for i in range(m):
        P = np.zeros((card,card))
        for j in range(card):
            P[:,j] = (1-lst_acc[i])/(card-1)
            P[j][j] = lst_acc[i]
            P[:,j] = P[:,j] * (1-lst_abstains[i])
        
        P = np.vstack((np.ones(card)*lst_abstains[i],P))
        lst_P.append(P)
    return lst_P


def run_ws_workflow(X_train, Y_train, conf):
    """

    Parameters
    ----------
    X_train
    Y_train
    conf

    Returns
    -------

    """
    d = conf['num_classes']
    n = len(X_train)
    out = {}
    r_utils = RankingUtils(d)
    Y_train_ranks = transformToRankedLabels(X_train,Y_train)
    
    if conf['ws_method']=='rankings':
        thetas = conf['thetas']
        
        print('got Y_train_ranks')
        Y_train_class = [Y_train_ranks[i][0] for i in range(len(Y_train_ranks))]
        #accuracy_score(Y_train_class,y_train)
        L = generate_ranking_LFs(Y_train_ranks, thetas)
        print('generated LFs, ', L.shape)
        
        
        lst_D = np.array([r_utils.get_pair_wise_dists(L[i, :, :]) for i in range(n)])
        print(lst_D.shape)
        wsr = WeakSupRanking(r_utils)
        wsr.train(conf, L)
        
        Y_train_ranks_tilde = wsr.inferRanking(conf, L, lst_D)
        out['parameter_error'] = np.linalg.norm(wsr.thetas-thetas)#/len(thetas)
        
   
        
    elif conf['ws_method']=='snorkel':
        lst_acc = conf['lst_acc']
        lst_abstains = conf['lst_abstains']
        
        lst_P = get_LF_class_probs(lst_acc,lst_abstains,d)
        L = generate_class_LFs(lst_P,Y_train)
        
        label_model = LabelModel(cardinality=d, verbose=True)
        label_model.fit(L_train=L, n_epochs=500, log_freq=100, seed=123)
        
        Y_train_class_tilde,probs  = label_model.predict(L, return_probs=True)
        
        Y_train_ranks_tilde = [np.argsort(-prob) for prob in probs]
        estimated_lst_P = list(label_model.parameters())[0]
        estimated_lst_P = estimated_lst_P.detach().numpy()
        lst_P = np.vstack([P[1:] for P in lst_P])
        out['parameter_error'] = np.linalg.norm(estimated_lst_P-lst_P)
        
        '''
        #label_model_acc = label_model.score(L=L, Y=np.array(Y), tie_break_policy="random")["accuracy"]
        #print(label_model_acc)
        '''

    Y_train_class_tilde = [Y_train_ranks_tilde[i][0] for i in range(n)]    
    mean_kt = r_utils.mean_kt_distance(Y_train_ranks, Y_train_ranks_tilde)
    out['mean_kt'] = mean_kt
    accc = evaluate(Y_train_ranks_tilde, Y_train, lst_k=[1, 2, 3])
    out['ac-1'] = accc[0]
    out['ac-2'] = accc[1]
    out['ac-3'] = accc[2]
    
    per_class_acc = confusion_matrix(Y_train, Y_train_class_tilde, normalize='true').diagonal()
    out['per_class_acc'] = per_class_acc
    y = np.bincount(Y_train)
    
    out['class_count'] = [(i, y[i]) for i in np.nonzero(y)[0]]
    
    return Y_train_class_tilde, out


def evaluate(Y_tilde_ranks, Y_class, lst_k=[1]):
    """
    evaluate class c
    Parameters
    ----------
    Y_tilde_ranks: estimated ranks
    Y_class: class
    lst_k: the list of paramter k for top_k_accuracy_score

    Returns
    -------

    """
    Y_tilde_scores = []

    for y in Y_tilde_ranks:
        y = list(y)
        y_tilde_scores = np.zeros(len(y))
        j = 1
        for i in y:
            y_tilde_scores[i] = 1/j
            j += 1
        Y_tilde_scores.append(y_tilde_scores)
    return [top_k_accuracy_score(Y_class, Y_tilde_scores, k=k) for k in lst_k]
    
def estimate_theta_acc(theta, d):
    """
    estimate the acurracy of parameter theta
    Parameters
    ----------
    theta
    d

    Returns
    -------

    """
    r_utils = RankingUtils(d)
    mlw = Mallows(r_utils, theta)
    y = r_utils.get_random_ranking()
    L = np.array([mlw.sample(y)[0] for i in range(4000)])
    L1 = np.array([L[i][0] for i in range(len(L))])
    L2 = np.array([L[i][1] for i in range(len(L))])
    L3 = np.array([L[i][2] for i in range(len(L))])
    z1 = 1*(L1 == y[0])
    z2 = 1*(L2 == y[0])
    z3 = 1*(L3 == y[0])
    
    return (np.mean(z1), np.mean(z2), np.mean(z3))

def build_dict_d_theta_mu(lst_d,lst_thetas,save_to_file_name):
    """
    build dictionary with d, theta, mu and save it as pickle
    Parameters
    ----------
    lst_d
    lst_thetas
    save_to_file_name

    Returns
    -------

    """
    dict_d_theta_mu = {}

    # already exist --> update
    if os.path.isfile(save_to_file_name):
        dict_d_theta_mu = pickle.load(open(save_to_file_name, 'rb'))
        
    for d in lst_d:
        dict_theta_mu = {}

        # save d
        if d in dict_d_theta_mu:
            dict_theta_mu = dict_d_theta_mu[d]

        # save theta and accuracy
        for theta in lst_thetas:
            key_theta = int(theta*100)
            if not key_theta in dict_theta_mu:
                print('estimating for ', d, theta)
                estimated_acc = estimate_theta_acc(theta, d)
                dict_theta_mu[key_theta]=estimated_acc
    # dump pickle
    pickle.dump(dict_d_theta_mu, open(save_to_file_name, 'wb'))
    return dict_d_theta_mu


def load_dict_d_theta_mu(load_from_file_name):
    """
    help function to load experiment parameters
    Parameters
    ----------
    load_from_file_name

    Returns
    -------

    """
    return pickle.load(open(load_from_file_name, 'rb'))