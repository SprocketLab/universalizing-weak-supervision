from synthetic_dataset import SyntheticRankingDataset
from movies_dataset import MoviesRankingDataset
from boardgames_dataset import BoardGamesRankingDataset

def create_dataset(data_conf):
    dataset_name = data_conf['dataset_name']
    if(dataset_name == 'movies'):
        return MoviesRankingDataset(data_conf)
    elif(dataset_name == 'boardgames'):
        return BoardGamesRankingDataset(data_conf)
    elif(dataset_name == 'synthetic'):
        return SyntheticRankingDataset(data_conf)
    else:
        raise NotImplemented