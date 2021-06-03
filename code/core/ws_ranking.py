from ws_lib import *
from mallows import *
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class WeakSupRanking:
    def __init__(self, r_utils):
        self.thetas = None
        self.r_utils = r_utils

   
    def train(self, conf, L, numLFs=None):
        """

        Parameters
        ----------
        conf: configuration dictionary, keys: 'train_method', 'inference_rule'
        L: labels, dim (n, m) where n is the number of examples and m is the number of LFs (numLFs)
        numLFs: the number of labeling functions

        Returns
        -------

        """ 
        n = len(L)  # the number of examples
        if numLFs is None:
            m = len(L[0])
        else:
            m = numLFs

        d = len(L[0][0])

        expected_dists = np.zeros(m)

        if conf['train_method'] == 'triplet' or conf['train_method'] == 'triplet_opt':
            # TODO: handle the case when num LF  is not multiple of 3

            i = 0
            order, D = self.order_LFs_on_correlation(L,m)
            
            #order = np.arange(m)
            #D = None

            # solve 3 variable systems iteratively
            while i < m-2:
                l1, l2, l3 = order[i], order[i+1], order[i+2]
                ac3 = self.solve_triplet(L, l1, l2, l3, D)
                expected_dists[l1] = ac3[0]
                expected_dists[l2] = ac3[1]
                expected_dists[l3] = ac3[2]

                i += 3

            if conf['train_method'] == 'triplet':
                self.thetas = 1.0 / np.array(expected_dists)

            elif conf['train_method'] == 'triplet_opt':
                mlw = Mallows(self.r_utils, 1.0)
                logger.info("expected_dists {}".format(expected_dists))
                self.thetas = np.array([mlw.estimate_theta(d, expected_dists[i])
                                        for i in range(len(expected_dists))])
                self.thetas = self.thetas.clip(1e-1, 100)

    def mean_distance(self, L, l1, l2, normalize=False):
        """
        Calculate mean distance of two distance, kendall tau distance is used as distance
        Parameters
        ----------
        L: labels, dim (n, m) where n is the number of examples and m is the number of LFs (numLFs)
        l1: label function 1
        l2: label function 2
        normalize

        Returns
        -------

        """
        n = len(L)

        mu = np.mean([self.r_utils.kendall_tau_distance(L[i][l1], L[i][l2], normalize=normalize)
                      for i in range(n)],axis=0)
        logger.info("mu {}".format(mu))
        return mu

    def order_LFs_on_correlation(self,L,numLFs=None):
        """
        Parameters
        ----------
        L: labels, dim (n, m) where n is the number of examples and m is the number of LFs (numLFs)

        Returns
        -------
        order: order of label functions based on their max distance
        D: the distance matrix of label functions
        """
        if(numLFs):
            m = numLFs
        else:
            m = len(L[0])  # the number of label functions
        dists = []  # the list of triplets (i, j, distance(i,j))
        D = np.zeros((m, m)) # the mean distance between label functions

        # fill out the distance matrix
        for i in range(m):
            for j in range(i):
                d_ij = self.mean_distance(L, i, j)
                dists.append((i, j, d_ij))
                D[i][j] = d_ij
                D[j][i] = d_ij

        dists = sorted(dists, key=lambda x: x[2], reverse=True)

        # order of label functions based on their max distance
        order = []
        for i, j, d in dists:
            if not i in order:
                order.append(i)
            if not j in order:
                order.append(j)

        return order, D

    def solve_triplet(self, L, l1, l2, l3, D=None):
        """

        Parameters
        ----------
        L: labels, dim (n, m) where n is the number of examples and m is the number of LFs (numLFs)
        l1: label function 1
        l2: label function 2
        l3: label function 3
        D: distance matrix of label functions

        Returns
        -------

        """
        if D is None:
            mu_12 = self.mean_distance(L, l1, l2)
            mu_23 = self.mean_distance(L, l2, l3)
            mu_31 = self.mean_distance(L, l3, l1)
        else:
            mu_12 = D[l1][l2]
            mu_23 = D[l2][l3]
            mu_31 = D[l3][l1]

        ac3 = solve_3_var_system_sum(mu_12, mu_23, mu_31)
        return ac3

    def infer_ranking(self,conf,L,numLFs=None,lst_D=None):
        """

        Parameters
        ----------
        conf: configuration dictionary, keys: 'train_method', 'inference_rule'
        L: labels, dim (n, m) where n is the number of examples and m is the number of LFs (numLFs)
        numLFs
        lst_D

        Returns
        -------

        """
        if numLFs is None:
            numLFs = len(L[0])

        k = numLFs

        Y_tilde = None
        n = len(L)
        
        # label inference based on kemeny rule
        if conf['inference_rule'] =='kemeny':
            if lst_D is None:
                Y_tilde = [self.r_utils.kemeny(L[i][:k]) for i in range(n)]
            else:
                Y_tilde = [self.r_utils.kemeny(L[i][:k],lst_D[i][:k, :k]) for i in range(n)]

        # label inference based on weighted kemeny rule
        elif conf['inference_rule'] == 'weighted_kemeny':
            if lst_D is None:
                Y_tilde = [self.r_utils.weighted_kemeny(L[i][:k], self.thetas[:k]) for i in range(n)]
            else:
                Y_tilde = [self.r_utils.weighted_kemeny(L[i][:k], self.thetas[:k], lst_D[i][:k,:k])
                           for i in range(n)]

        # label inference based on pairwise majority
        elif conf['inference_rule'] == 'pairwise_majority':
            Y_tilde = [self.r_utils.pair_wise_majority(L[i][:k], weights=None) for i in range(n)]

        # label inference based on weighted pairwise majority
        elif conf['inference_rule'] == 'weighted_pairwise_majority':
            Y_tilde = [self.r_utils.pair_wise_majority(L[i][:k], weights=self.thetas[:k]) for i in range(n)]

        # label inference based on position estimation
        elif conf['inference_rule'] == 'position_estimation':
            Y_tilde = [self.r_utils.pos_est_majority(L[i][:k], weights=None) for i in range(n)]

        # label inference based on weighted position estimation
        elif conf['inference_rule']=='weighted_position_estimation':
            Y_tilde = [self.r_utils.pos_est_majority(L[i][:k], weights=self.thetas[:k]) for i in range(n)]

        h = [y.mask_items([]) for y in Y_tilde]

        return Y_tilde