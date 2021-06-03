from abc import ABC, abstractmethod
 
class AbstractRankingLF(ABC):
    '''
    lst_feature_map :  list of dictionaries, each dictornary
                       contains feature_name: feature_value
    return: ranking on indices from 0 to len(lst_feature_map)-1
    '''
    @abstractmethod
    def apply(self,lst_feature_map):
        pass


class AbstractRegressionLF(ABC):
    """
    LF abstraction for regression label function
    """
    @abstractmethod
    def apply(self, df):
        pass