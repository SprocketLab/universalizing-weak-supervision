from .base_lf import *
from ranking_utils import * 
class FeatureRankingLF(AbstractRankingLF):
    def __init__(self,rank_on_feature, d, highest_first=True ):
        self.rank_on_feature = rank_on_feature 
        self.highest_first = highest_first
        self.r_utils = RankingUtils(d)

    def apply(self,lst_feature_map):
        k = len(lst_feature_map)
        lst_id_f_val = [(i,lst_feature_map[i][self.rank_on_feature]) for i in range(k) ]
        out = sorted(lst_id_f_val, key = lambda x: x[1], reverse=self.highest_first)
        out = Ranking([i for i,v in out], self.r_utils)
        return out

    def apply_mat(self, X):
        L = []
        k = X.shape[1]
        for row in range(X.shape[0]):
            lst_id_f_val = [(i, X[row][i][self.rank_on_feature]) for i in range(k)]
            out = sorted(lst_id_f_val, key=lambda x: x[1], reverse=self.highest_first)
            out = Ranking([i for i, v in out], self.r_utils)
            L.append(out)
        return L
