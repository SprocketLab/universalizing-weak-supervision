from collections import defaultdict
from ranking_utils import RankingUtils
from mallows import Mallows
from ws_ranking import WeakSupRanking
import numpy as np
import random
from math import ceil

def get_mask(y, m, p):
    """
    sampling mask based on
    Parameters
    ----------
    y: an instance of ranking
    m: the number of labeling functions
    p: the fraction of unmasked items

    Returns
    -------

    """
    d = len(y)
    k = m-ceil(p*m)
    masks = [[] for _ in range(m)]
    y = y.permutation

    # mask random sampling
    for x in y:
        lfs = random.sample(range(m), k=k)  # use weights here
        for lf in lfs:
            masks[lf].append(x)

    # guarantee there are at least two items
    for i in range(m):
        if (len(masks[i]) > d-2):
            masks[i] = masks[i][2:]
    return masks 

def apply_masks(Y, L, m, p, d_mask=None):
    """
    apply masks to weak labels L
    Parameters
    ----------
    Y: True labels
    L: Weak labels without masks
    m: The number of labeling functions
    p: The fraction of unmasked items
    d_mask: alternative of p, mask except d_mask items

    Returns
    -------

    """
    for i in range(len(Y)):
        if(p is not None):
            masks = get_mask(Y[i], m, p)
            for j in range(m):
                if(len(masks[j])>0):
                    L[i][j].mask_items(masks[j]) 
        else:
            for j in range(m):
                L[i][j].mask_items(L[i][j][-d_mask:])

def sample_mallows_LFs(Y, m, thetas, p=None, d_mask=None):
    """
    sample weak labels and
    Parameters
    ----------
    Y: True labels
    m: The number of labeling functions
    thetas: the list of theta in mallows model
    p: the fraction of unmasked items
    d_mask: alternative of p, mask except d_mask items

    Returns
    -------

    """

    d = len(Y[0])

    thetas = np.array(thetas)
    r_utils = RankingUtils(d)

    # get LFs
    lst_mlw = [ Mallows(r_utils,theta) for theta in thetas ]
    L = [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y]

    if (p is not None and p >= 0.05):
        # mask and return
        apply_masks(Y, L, m, p)
    elif (d_mask is not None and d_mask >0):
        apply_masks(Y, L, m, None, d_mask)
    return L 
    
def estimate_theta(L):
    d =  len(L[0][0])
    m = len(L[0])
    r_utils = RankingUtils(d)
    wsr = WeakSupRanking(r_utils)
    conf = {"train_method":"triplet_opt"}
    wsr.train(conf, L, m)
    return wsr.thetas 

def get_pair_wise_dists(L):
   # Pre Compute distances
    n = len(L)
    d =  len(L[0][0])
    r_utils = RankingUtils(d)
    lst_D = [r_utils.get_pair_wise_dists(L[i]) for i in range(n)]
    return lst_D

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
