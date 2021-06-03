
import sys

sys.path.append('../../')
sys.path.append('../../core')

from collections import defaultdict 

from ranking_utils import *
from mallows import *
from ws_ranking import *
import numpy as np
import matplotlib.pyplot as plt
from ranking_digraph import *
from math import ceil

def get_mask(y,m,p):
    d = len(y)
    k = m-ceil(p*m)
    for i in range(1):
        masks = [[] for i in range(m)]
        y = y.permutation
        for x in y:
            lfs = random.sample(range(m),k=k)  # use weights here
            for lf in lfs:
                masks[lf].append(x)
        for i in range(m):
            if(len(masks[i])>d-2):
                masks[i] = masks[i][2:]
    
    return masks 

def apply_masks(Y,L,m,p,d_mask=None):
    
    for i in range(len(Y)):
        if(p is not None):
            masks = get_mask(Y[i],m,p)
            for j in range(m):
                if(len(masks[j])>0):
                    L[i][j].mask_items(masks[j]) 
        else:
            for j in range(m):
                L[i][j].mask_items(L[i][j][-d_mask:])
    
def generate_true_rankings(d,n):
    r_utils = RankingUtils(d)
    Y = [r_utils.get_random_ranking() for i in range(n)]
    return Y

def sample_mallows_LFs(Y,m,thetas,p=None,d_mask=None):
    # p = None means sample full rankings
    # else each alternative will appear in at least p fraction of LFs.
    n = len(Y)
    d = len(Y[0])

    thetas = np.array(thetas)
    r_utils = RankingUtils(d)

    # get LFs
    lst_mlw = [ Mallows(r_utils,theta) for theta in thetas ]
    L = [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y]

    if(p is not None and p>=0.05):
        # mask and return
        apply_masks(Y,L,m,p)
    elif(d_mask is not None and d_mask >0):
        apply_masks(Y,L,m,None,d_mask)

    
    return L 
    
def estimate_theta(L):
    d =  len(L[0][0])
    m = len(L[0])
    r_utils = RankingUtils(d)
    wsr = WeakSupRanking(r_utils)
    conf = {"train_method":"triplet_opt"}
    wsr.train(conf,L,m)
    return wsr.thetas 

def get_pair_wise_dists(L):
   # Pre Compute distances
    n = len(L)
    d =  len(L[0][0])
    r_utils = RankingUtils(d)
    lst_D = [r_utils.get_pair_wise_dists(L[i]) for i in range(n)]
    return lst_D

def run_kemeny_inferences(Y,L,theta_star,theta_hat,m,lst_D):
    
    n = len(L)
    max_m = len(L[0])
    d =  len(L[0][0])
    
    r_utils = RankingUtils(d)
    wsr = WeakSupRanking(r_utils)

    out = {}

    # unweighted aggregation
    conf = {"inference_rule":"kemeny"}
    Y_tilde = wsr.infer_ranking(conf,L,m,lst_D)
    out['mean_kt_unwtd_kemeny'] = r_utils.mean_kt_distance(Y, Y_tilde)
    
    # use optimal theta
    conf = {"inference_rule":"weighted_kemeny"}
    wsr.thetas = np.array(theta_star)[:m]
    Y_tilde = wsr.infer_ranking(conf,L,m,lst_D)
    out['mean_kt_wtd_kemeny_opt'] = r_utils.mean_kt_distance(Y, Y_tilde)

    # use estimated theta
    wsr.thetas = theta_hat[:m]
    Y_tilde = wsr.infer_ranking(conf,L,m,lst_D)
    out['mean_kt_wtd_kemeny_hat'] = r_utils.mean_kt_distance(Y, Y_tilde)

    return out 

def run_partial_ranking_inferences(Y,L,theta_star,theta_hat,m):
    n = len(L)
    max_m = len(L[0])
    d =  len(L[0][0])
    r_utils = RankingUtils(d)
    wsr = WeakSupRanking(r_utils)
    out  = {}

     # unweighted aggregation
    conf = {"inference_rule":"pairwise_majority"}
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_unwtd_pma'] = r_utils.mean_kt_distance(Y, Y_tilde)

    conf = {"inference_rule":"position_estimation"}
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_unwtd_pos'] = r_utils.mean_kt_distance(Y, Y_tilde)

    conf = {"inference_rule":"weighted_pairwise_majority"}
    wsr.thetas = theta_star
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_wtd_pma_opt'] = r_utils.mean_kt_distance(Y, Y_tilde)

    wsr.thetas = theta_hat
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_wtd_pma_hat'] = r_utils.mean_kt_distance(Y, Y_tilde)

    conf = {"inference_rule":"weighted_position_estimation"}
    wsr.thetas = theta_star
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_wtd_pos_opt'] = r_utils.mean_kt_distance(Y, Y_tilde)

    wsr.thetas = theta_hat
    Y_tilde = wsr.infer_ranking(conf,L,m)
    out['mean_kt_wtd_pos_hat'] = r_utils.mean_kt_distance(Y, Y_tilde)

    return out 

def collect_outs(lst_out):
    out = defaultdict(list)
    for o in lst_out:
        for k in o.keys():
            out['lst_'+k].append(o[k])
    return out 

def mean_std( out, axis=0):
    o = {}
    for k in out.keys():
        o['mean_{}_{}'.format(axis,k)]= np.mean(out[k],axis=axis)
        o['std_{}_{}'.format(axis,k)]= np.std(out[k],axis=axis)
    return o
