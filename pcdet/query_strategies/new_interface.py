from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch.nn as nn

class StrategyNewInterface(ABC):
    def __init__(self, *args, **kwarg):
        pass
    
    @overload
    @staticmethod
    @abstractmethod
    def query(model : nn.Module, handler: ModelHandler, dataset_handler: DataSetHandler,training_config: Dict, query_size: int, device, *arg, **kwargs) -> Tuple[List, List]:
        pass

class ModelHandler():
    def get_grad_embedding(self, model, probs, feat):
        pass
    
    def enable_dropout(self, model):
        pass  

class DataSetHandler():  
    unlabeled_idcs : List[int]  
    labeled_idcs : List[int]  
      
    def get_unlabeled_loader(self):  
        pass  
      
    def get_labeled_loader(self):  
        pass 