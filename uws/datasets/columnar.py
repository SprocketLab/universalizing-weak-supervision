"""A thin wrapper around a CSV of items with features and a label column."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


class BasicColumnarDataset:
    """Items (rows) keyed by ``id_feature``; duplicates of an id are dropped (first kept)."""

    def __init__(self, file_path, id_feature: str, features_subset: Optional[Sequence[str]] = None,
                 label_feature: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        self.df = pd.read_csv(file_path) if df is None else df.copy()
        self.df = self.df.drop_duplicates(subset=[id_feature]).reset_index(drop=True)
        self.feature_subset = list(features_subset) if features_subset is not None else list(self.df.columns)
        self.features = list(self.df.columns)
        self.label_feature = label_feature
        self.id_feature = id_feature
        self.hash_id2idx = {v: i for i, v in enumerate(self.df[id_feature])}

    def preprocess(self, fill_na_by=0, normalize: bool = True):
        """Fill missing values and scale every feature of ``feature_subset`` by its max."""
        self.df = self.df.fillna(fill_na_by)
        if normalize:
            for feature in self.feature_subset:
                mx = self.df[feature].max()
                if mx:
                    self.df[feature] = self.df[feature] / mx

    def get_feature_map(self, id_val, drop_label_feature: bool = True, drop_id_feature: bool = True) -> dict:
        d = self.df.iloc[self.hash_id2idx[id_val]][self.feature_subset].to_dict()
        if drop_label_feature and self.label_feature:
            d.pop(self.label_feature, None)
        if drop_id_feature:
            d.pop(self.id_feature, None)
        return d

    def get_ref_map(self, id_val) -> dict:
        return self.df.iloc[self.hash_id2idx[id_val]].to_dict()

    def get_feature(self, id_val, feature):
        return self.df.iloc[self.hash_id2idx[id_val]][feature]

    def get_label(self, id_val):
        assert self.label_feature, "label_feature is not set"
        return self.get_feature(id_val, self.label_feature)

    def get_all_key_values(self):
        return self.df[self.id_feature].unique()
