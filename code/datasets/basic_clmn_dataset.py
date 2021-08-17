import pandas as pd
import numpy as np 

class BasicColumnarDataset:
    def __init__(self,file_path,id_feature,features_subset=None,label_feature=None):
        self.df = pd.read_csv(file_path)
        self.df.drop_duplicates(subset=[id_feature], inplace = True)
        self.feature_subset = features_subset
        self.features = list(self.df.columns)
        self.label_feature = label_feature
        self.id_feature = id_feature

        # hashing for fast searching
        self.df = self.df.reset_index(drop=True)
        self.hash_id2idx = {}
        for id_val in pd.unique(self.df[id_feature]):
            self.hash_id2idx[id_val] = int(self.df[self.df[id_feature] == id_val].index[0])

    def preprocess(self, fill_na_by=0, normalize=True):
        """
        preprocess dataframe
        1. fill na
        2. normalize each feature with its max value (it could be unnecessary now)
        Parameters
        ----------
        df
        lf_features
        normalize

        Returns
        -------

        """
        self.df = self.df.fillna(fill_na_by)
        if normalize:
            for feature in self.feature_subset:
                self.df[feature] = self.df[feature] / self.df[feature].max()

    def get_feature_map(self,id_val,drop_label_feature=True,drop_id_feature=True):
        d = self.df.iloc[self.hash_id2idx[id_val]][self.feature_subset].to_dict()

        if(drop_label_feature and self.label_feature):
            if(self.label_feature in d):
                d.pop(self.label_feature)
            if(self.id_feature in d):
                d.pop(self.id_feature)

        return d
    
    def get_ref_map(self,id_val):
        d = self.df.iloc[self.hash_id2idx[id_val]].to_dict()
        return d 

    def get_feature(self,id_val,feature):
        """
        get specific feature from specific id_val
        Parameters
        ----------
        id_val
        feature

        Returns
        -------

        """
        y = self.df.iloc[self.hash_id2idx[id_val]].to_dict()[feature]
        return y
    
    def get_label(self,id_val):
        assert self.label_feature 
        return self.get_feature(id_val,self.label_feature)

    def get_all_key_values(self):
        return self.df[self.id_feature].unique()

    def df(self):
        return self.df 

