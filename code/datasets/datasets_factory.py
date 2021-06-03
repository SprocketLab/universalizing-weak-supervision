from imdb_tmdb_dataset import IMDBTMDBRankingDataset
from synthetic_dataset import SyntheticRankingDataset
from boardgame_dataset import BoardGameRankingDataset

def create_dataset(data_conf):
    dataset_name = data_conf['dataset_name']
    if(dataset_name == 'imdb_tmdb'):
        return IMDBTMDBRankingDataset(data_conf)
    elif(dataset_name == 'board-games'):
        return BoardGameRankingDataset(data_conf)
    elif(dataset_name == 'synthetic'):
        return SyntheticRankingDataset(data_conf)
    else:
        raise NotImplemented