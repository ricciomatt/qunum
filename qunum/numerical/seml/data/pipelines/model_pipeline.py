import torch
from typing import Callable, Iterator
import copy
class ModelPipeline:
    def __init__(self, 
                 Models:tuple[Callable],
                 training:dict[bool] = None):
        if(training is None):
            training = {}
            default = True
        elif(type(training) is False):
            default = training
            training = {}
        else:
            default = True
        
        self.Models = tuple(Models)
        for m in self.Models:
            if(m not in training):
                training[m] = copy.copy(default)
        self.training = training
        return
    
    def __call__(self, x):
        for M in self.Models:
            with torch.no_grad():
                x = M(x)
        return x
    
    def __len__(self,):
        return len(self.Models)
    
    def __getitem__(self, idx):
        return self.Models[idx]
    
    def append(self, M):
        Mod = list(self.Models)
        Mod.append(M)
        self.Models = tuple(Mod)
        return