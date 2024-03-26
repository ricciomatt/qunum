from typing import Callable
import numpy as np
import torch
class ModelTracker:
    def __init__(self,
                 functions:dict[str:Callable],
                 track_loss:bool=True,
                 )->None:
        if(track_loss):
            functions['loss']=None
        self.functions = functions
        self.metrics = dict.fromkeys(functions.keys(), [])
        self.ix = 0
        return
    
    def __call__(self,y:torch.Tensor, yh:torch.Tensor, L:torch.Tensor):
        self.metrics['loss'].append(L.cpu().detach().numpy())
        for f in self.functions:
            if(f != 'loss'):
                self.metrics[f].append(self.functions[f](yh.cpu(),y.cpu()))
        
        self.ix+=1
        return