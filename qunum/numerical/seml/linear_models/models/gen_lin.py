from .out_functions import getMap, Id
from typing import Iterable, Any, Sequence
from torch.optim import Adam, SGD, Optimizer
from .linear_model import LinearNN
from ..estimators import *
from torch import Tensor
from ...fit.grad_descent.object import GradDescentTrain
from ...data.data_loaders import DataLoader
from ...data.preprocessing import ComputePoly
import torch 
import numpy as np 
opt_map = {'adam':Adam, 'sgd':SGD}
options = {'GLS', 'Ridge', 'Quantile', 'Lasso'}
def doNothing(x:Tensor)->Tensor:
    return x
class GenLin:
    def __init__(self, *args, 
                 paramter_estimator:str = 'GLS', 
                 F:Id = Id(),
                 x_shp:int|None = None,
                 y_shp:int|None = None,
                 x_cols:Sequence[str]|None=None,
                 y_cols:Sequence[str]|None=None,
                 interaction_cols:Sequence[str|int]|None = None,
                 lamda_or_quantile:float = 1.,
                 lr:float = 1e-2,
                 order:int = 1, 
                 interaction_order:int = 0,
                 optimizer:str = 'adam', 
                 num_epochs:int = int(1e3),
                 device:str = 'cpu',
                 poly_pipe:bool = False,
                 **kwargs)->None|bool:
         
        if(x_shp is None and x_cols is None):
            raise ValueError('Must provide an input shape or input column list')
        elif(x_shp is not None and x_cols is None):
            x_cols = np.arange(x_shp)
        else:
            x_shp = len(x_cols)
        if(y_shp is None and y_cols is None):
            raise ValueError('Must provide an output shape or output column list')
        elif(y_shp is not None and y_cols is None):
            y_cols = np.arange(y_shp)
        else:
            y_shp = len(y_cols)
        self.order = order 
        self.interaction_order = interaction_order
        poly = ComputePoly(ipt_cols= x_cols, 
                    order=order, 
                    interaction_order=interaction_order, 
                    interaction_cols=interaction_cols)
        if(poly_pipe): self.poly = poly
        else: self.poly = doNothing 
        self.model = LinearNN(poly.I, y_shp, F)
        self.lamda_or_quantile = lamda_or_quantile
        self.device = device
        if(paramter_estimator not in options):
            raise ValueError(f'Only implemented for {str(options)}')
        elif(paramter_estimator in {'GLS', 'Ridge'}):
            self.Loss = torch.nn.MSELoss()
            self.fit_it = mapping[paramter_estimator]
            self.grad_ = False
            self.Optim = None
            
        else:
            self.Loss = mapping[paramter_estimator](lam = self.lamda_or_quantile)
            self.grad_ = True
            if(isinstance(optimizer, str)):
                optimizer = opt_map[optimizer](lr=lr, **kwargs)
            self.Optim = optimizer
            self.fit_it = GradDescentTrain(self.model, self.Loss, self.Optim, epochs=num_epochs, device= device)
        return  
    
    def change_optim(self, opt:Optimizer, *args, **kwargs):
        self.Optim = opt(self.model.parameters, *args, **kwargs)
        return
    
    def train(self, *args:tuple['x':torch.Tensor, 'y':torch.Tensor, 'xval':torch.Tensor, 'yval':torch.Tensor,]|tuple['trn':DataLoader, 'val':DataLoader], **kwargs:dict[Any])->None:
        self.model.requires_grad_(True)
        if(self.Optim is not None):
            self.fit_it(self.poly(args[0]),*args[1:], **kwargs)
        else:
            self.model.parameters().data[0] = self.fit_it(self.poly(args[0]),args[1],self.lamda_or_quantile)
        self.model.requires_grad_(False)
        return
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        return self.model(self.poly(x))
        