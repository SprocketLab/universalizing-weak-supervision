from .base_lf import AbstractRegressionLF
import pandas as pd
import numpy as np

class ContinuousLF(AbstractRegressionLF):
    def __init__(self, feature_names):
        if type(feature_names) == str:
            feature_names = [feature_names]
        self.feature_names = feature_names

    def apply(self, df, logarithmic=False, default_val=6, y_min=0, y_max=10):
        L = pd.DataFrame()
        for col in self.feature_names:
            if col not in df.columns:
                print(col, "doest not exist in df!")
                continue
            feature_col = df[col]

            if logarithmic:
                feature_col = np.log(feature_col + 1)

            # init
            weak_label = feature_col

            # scaling to the range of rating
            weak_label = np.clip(weak_label / weak_label.quantile(0.98) * 10, y_min, y_max)

            # handling zero values with the predetermined default_val
            weak_label.loc[weak_label == 0] = default_val

            L[col] = weak_label
        return L


class DiscreteLF(AbstractRegressionLF):
    def __init__(self, feature_names, label_feature):
        if type(feature_names) == str:
            feature_names = [feature_names]
        self.feature_names = feature_names
        self.label_feature = label_feature

    def apply(self, df, debug=False):
        L = pd.DataFrame()
        for col in self.feature_names:
            if col not in df.columns:
                print(col, "doest not exist in df!")
                continue

            groupby_mean = df.groupby(col).mean()[self.label_feature]
            if debug:
                print(groupby_mean)
            weak_label = df[[col]].merge(groupby_mean, how='left', on=col).drop(col, axis=1)
            L[col] = np.ndarray.flatten(weak_label.values)
        return L
