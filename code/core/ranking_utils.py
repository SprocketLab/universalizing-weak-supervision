import random
import copy
import numpy as np
import itertools
from scipy.stats import kendalltau
from collections.abc import Sequence
from ranking_digraph import RankingDiGraph
from collections import defaultdict
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class RankingUtils:
    def __init__(self, d):
        """
        self.items: the list of items
        self.unique_pairs: the list of all pairs of items
        self.pair_index_map: the mapping pairs to a number

        Parameters
        ----------
        d: cardinality (the number of items)
        """
        self.d = d
        self.items = list(range(d))
        self.unique_pairs = []
        for i in range(d):
            for j in range(i+1, d):
                self.unique_pairs.append((i, j))

        self.pair_index_map = {}
        k = 0
        for i in range(d):
            for j in range(i+1, d):
                self.pair_index_map[self.pair_key(self.items[i], self.items[j])] = k
                k += 1
        self.dist_counts = None


    def get_unique_pairs(self):
        return self.unique_pairs

    def get_random_ranking(self):
        """
        generate random ranking from self.items
        Returns
        -------
        s: a random ranking of d items
        """
        s = copy.deepcopy(self.items)
        random.shuffle(s)
        s = Ranking(s, self)
        return s

    def ranking_to_pair_signs(self, r):
        """

        Parameters
        ----------
        r: Ranking object

        Returns
        -------
        z: d(d-1)/2 dim, pair signs, for "i,j", if r[i] < r[j] --> 1, otherwise -1 (and then masked)
        """
        u = len(r)*(len(r)-1)//2
        z = np.zeros(u)
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                a, b = r[i], r[j]
                pk = self.pair_key(a, b)
                if pk in self.pair_index_map:
                    idx = self.pair_index_map[self.pair_key(a, b)]
                    z[idx] = 1 * r.mask[idx]
                else:
                    idx = self.pair_index_map[self.pair_key(b, a)]
                    z[idx] = -1 * r.mask[idx]
        return z

    def pair_key(self, a, b):
        """
        Parameters
        ----------
        a: item 1
        b: item 2

        Returns
        -------

        """
        return str(a)+','+str(b)

    def Z(self, r1, r2):
        """
        element-wise multiply sign pairs of two rankings
        Parameters
        ----------
        r1
        r2

        Returns
        -------
        z1 * z2: (d(d-1)/2, 1) dim
        """
        if isinstance(r1, Ranking):
            z1 = r1.z
        else:
            z1 = self.ranking_to_pair_signs(r1)

        if isinstance(r2, Ranking):
            z2 = r2.z
        else:
            z2 = self.ranking_to_pair_signs(r2)

        return np.multiply(z1, z2)

    def kendall_tau_distance(self, r1, r2, normalize=False):
        """
        calculate kendall's tau distance

        Parameters
        ----------
        r1
        r2
        normalize

        Returns
        -------
        """
        D = np.zeros(len(r1.z))
        #p = 1.0/len(r1) #len(z)
        #p = 1.0/len(z)
        p = 1e-1 #1 # 0.5 #1e-2

        n = 0
        for i in range(len(D)):
            if not (r1.z[i] == 0 and r2.z[i] == 0):
                c = r1.z[i] * r2.z[i]
                if c == -1:
                    D[i] = 1
                elif c == 0:
                    D[i] = p
                n += 1
            else:
                D[i] = 1.0/len(D)

        d = sum(D)
        if normalize:
            d = float(d) /n
        return d

    def get_pair_wise_dists(self, lstRanks):
        """
        Parameters
        ----------
        lstRanks: the list of Ranking objects (size p)

        Returns
        -------
        the matrix of pairwise kendall tau distance ((p, p) dim)
        """
        p = len(lstRanks)
        D = np.zeros((p,p))
        for i in range(p):
            for j in range(i+1,p):
                d = self.kendall_tau_distance(lstRanks[i], lstRanks[j])
                D[i][j] = d
                D[j][i] = d
        return D

    def kemeny(self, lstRanks, D=None):
        """
        Get the aggregated rank based on kemeny rule
        The aggregated rank minimizes the mean distance with other rankings
        Parameters
        ----------
        lstRanks: the list of Ranking objects (size p)
        D: pre-calculated pairwise distance matrix

        Returns
        -------

        """
        if D is None:
            D = self.get_pair_wise_dists(lstRanks)

        logger.debug("D.shape {}".format(D.shape))
        mu = D.mean(axis=1)
        idx = np.argmin(mu)
        return lstRanks[idx]

    def pos_est_majority(self, lstRanks, weights=None):
        """
        position based majority votes

        Parameters
        ----------
        lstRanks: the list of Ranking objects
        weights: weights of each ranks

        Returns
        -------

        """
        # get the sum of signs (unweighted and weighted version)
        if weights is None:
            z_agg = sum([r.z for r in lstRanks])
        else:
            z_agg = sum([lstRanks[i].z*weights[i] for i in range(len(lstRanks))])

        # re-coding the signs based on z_agg
        z = []
        for x in z_agg:
            if x == 0: # tie
                z.append(0)
            elif x > 0: # win
                z.append(1)
            elif x < 0: # lose
                z.append(-1)
        positions = defaultdict(list)
        logger.debug("z_agg {}".format(z_agg))

        # calculate how many times each items lose or tie
        for i in self.items:
            l = 0 # the lost or tie count of item i
            for j in self.items:
                if i < j:
                    pk = self.pair_index_map[self.pair_key(i, j)]
                    if z[pk] == -1 or z[pk] == 0:
                        l += 1
                elif i > j:
                    pk = self.pair_index_map[self.pair_key(j, i)]
                    if z[pk] == 1 or z[pk] == 0:
                        l += 1
            positions[l].append(i)
        pi_hat = []
        logger.debug("positions {}".format(positions))

        # make a rank based on the count of lose or tie
        for k in sorted(positions.keys()):
            if len(positions[k]) == 0:
                continue
            if len(positions[k]) == 1:
                pi_hat.append(positions[k][0])
            else: # ties on the same lose or tie count --> randomly shuffle
                random.shuffle(positions[k])
                pi_hat.extend(positions[k])

        left = [x for x in self.items if x not in pi_hat]
        logger.debug("left {}".format(left))
        random.shuffle(left)
        pi_hat = pi_hat + left

        return Ranking(pi_hat, self)


    def pair_wise_majority(self, lstRanks, weights=None):
        """

        Parameters
        ----------
        lstRanks: the list of ranking
        weights: the weights of each ranking

        Returns
        -------

        """
        if weights is None:
            z_agg = sum([r.z for r in lstRanks])
            confidence = sum([np.abs(r.z) for r in lstRanks])
        else:
            z_agg = sum([lstRanks[i].z * weights[i] for i in range(len(lstRanks))])
            confidence = sum([np.abs(lstRanks[i].z*weights[i]) for i in range(len(lstRanks))])

        # re-coding the signs based on z_agg
        z  = []
        for x in z_agg:
            if x == 0:
                z.append(0)
            elif x > 0:
                z.append(1)
            elif x < 0:
                z.append(-1)

        return self.z_to_ranking(z,z_agg, confidence)

    def z_to_ranking(self, z, w=None, confidence=None):
        """
        make a diagraph with z (pair signs), and then make a new Ranking
        Parameters
        ----------
        z
        w
        confidence

        Returns
        -------

        """
        rdag = RankingDiGraph()
        rdag.nodes = copy.deepcopy(self.items)
        if w is None:
            w = np.zeros(len(z))
        if confidence is None:
            confidence = np.zeros(len(z))

        # add edges
        for i in range(len(z)):
            u, v = self.unique_pairs[i]
            if z[i] == 1:
                rdag.add_edge(u, v, w[i], confidence[i])
            elif z[i] == -1:
                rdag.add_edge(v, u, abs(w[i]), confidence[i])

            elif z[i] == 0 and confidence[i] > 0:
                rdag.add_edge(u, v, w[i], confidence[i])
                rdag.add_edge(v, u, w[i], confidence[i])

        permutation = rdag.topo_sort()
        return Ranking(permutation, self, z)

    def weighted_kemeny(self, lstRanks, weights, D=None):
        """
        weighted kenemy ranking
        Parameters
        ----------
        lstRanks
        weights
        D

        Returns
        -------

        """
        if D is None:
            D = self.get_pair_wise_dists(lstRanks)
        D_th = np.dot(D, np.diag(weights))
        mu = D_th.mean(axis=1)
        idx = np.argmin(mu)
        return lstRanks[idx]

    def mean_kt_distance(self, Y1, Y2, normalize=True):
        """
        mean kendall tau distance
        Parameters
        ----------
        Y1: the list of Ranking objects
        Y2: the list of Ranking objects
        normalize

        Returns
        -------

        """
        return np.mean([self.kendall_tau_distance(Y1[i], Y2[i], normalize) for i in range(len(Y1))])

    def build_dist_counts(self):
        """
        Calculate the number of distance starting with i
        Returns
        -------

        """
        d = len(self.items)
        p = (d*(d-1))//2
        D = np.zeros((d+2, p+1))

        D[2][0] = 1
        D[2][1] = 1
        for i in range(3, d + 1):
            prev = 0
            k = (i * (i - 1)) // 2
            for j in range(k+1):
                if j < i:
                    D[i][j] = prev + D[i-1][j]
                else:
                    D[i][j] = prev + D[i-1][j] - D[i-1][j-i]
                prev = D[i][j]
        return D

    def get_dist_counts(self):
        """
        get the number of distance starting with i
        Returns
        -------

        """
        if self.dist_counts is None:
            self.dist_counts = self.build_dist_counts()
        d = len(self.items)
        return self.dist_counts[d, :]

    def set_perm2int_int2perm_mapping(self):
        """
        Make perm2int and int2perm mapping to convert between permutation <--> integer
        Returns
        -------

        """
        values = [str(i) for i in range(self.d)]
        full_perms = ["".join(list(perm)) for perm in itertools.permutations(values)]
        perm2int_map = {}
        int2perm_map = {}
        for i, perm in enumerate(full_perms):
            perm2int_map[perm] = i
            int2perm_map[i] = [int(num) for num in perm]
        return perm2int_map, int2perm_map

    def perm2int(self, lstRanks):
        """
        Convert the list of permutations into the list of corresponding integers
        Parameters
        ----------
        lstRanks

        Returns
        -------

        """
        lstInts = [self.perm2int_map["".join([str(idx) for idx in r.permutation])] for r in lstRanks]
        return lstInts

    def int2perm(self, lstInts):
        """
        Convert the list of integers into the list of corresponding permutations
        Parameters
        ----------
        lstInts

        Returns
        -------

        """
        lstRanks= [Ranking(self.int2perm_map[i], r_utils=self) for i in lstInts]
        return lstRanks

class Ranking(Sequence):
    def __init__(self, permutation, r_utils=None, z=None):
        """

        Parameters
        ----------
        permutation: the permutation of items
        r_utils: RankingUtils object
        z: the list of pair signs
        """
        self.permutation = permutation

        if r_utils is None:
            self.r_utils = RankingUtils(len(permutation))
        else:
            self.r_utils = r_utils
        self.mask = np.ones(len(self.r_utils.unique_pairs))

        if z is None:
            self.z = self.r_utils.ranking_to_pair_signs(self)
        else:
            self.z = z

        super().__init__()

    def __len__(self):
        return len(self.permutation)

    def __getitem__(self, i):
        return self.permutation[i]

    def __str__(self):
        return str(self.permutation)

    def reverse(self):
        """
        reverse ranking
        Returns
        -------

        """
        p = list(reversed(self.permutation))
        return Ranking(p, self.r_utils)

    def mask_items(self, lst_items):
        """
        generate self.mask to mask not included items of self.permutation in lst_items
        Parameters
        ----------
        lst_items: the list of items

        Returns
        -------

        """
        d = len(self.permutation)
        pair_index_map = self.r_utils.pair_index_map

        # generate mask for lst_items
        for i in range(d):
            for j in lst_items:
                if i < j:
                    pk = self.r_utils.pair_key(i, j)
                    self.mask[pair_index_map[pk]] = 0
                elif i > j:
                    pk = self.r_utils.pair_key(j, i)
                    self.mask[pair_index_map[pk]] = 0

        # lst1: not included in lst_items
        lst1 = [i for i in self.permutation if i not in lst_items]

        # re-calculate pair signs based on mask
        self.z = self.r_utils.ranking_to_pair_signs(self)
        random.shuffle(lst_items)
        self.permutation = lst1 + lst_items

        # self.mask_permutation: zero mask based on whether included or not
        self.mask_permutation = [[0]*len(lst1)] + [[1]*len(lst_items)]
