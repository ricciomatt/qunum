from typing import Callable
import numpy as np
import torch
from plotly import graph_objects as go
def unity(x):
    return x 
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
    
    def plot_fun(self, metrics:list[str]=['loss'], funs:list[Callable]=[np.log]):
        data = list(map(lambda m: {
                'type':'scatter',
                'y':m[1](np.array(self.metrics[m[0]])),
                'x':np.arange(len(self.metrics[m[0]])),
                'name':m[0],
            }, zip(metrics,funs)))
        return go.Figure(data=data, layout={'title':'Loss Function', 'yaxis':{'title':'metric'},'xaxis':{'title':'step'}, 'height':500, 'width':1250})