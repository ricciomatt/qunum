from .out_functions import getMap, Id
from typing import Iterable, Any
from torch.optim import Adam, SGD, Optimizer
from .linear_model import LinearNN
from ..estimators import *
from torch import Tensor
from ...fitting_algos.grad_descent.object import GradDescentTrain
from ...data.data_loaders import DataLoader
import torch 
opt_map = {'adam':Adam, 'sgd':SGD}
options = {'GLS', 'Ridge', 'Quantile', 'Lasso'}
class GenLin:
    def __init__(self, *args, 
                 paramter_estimator:str = 'GLS', 
                 F:Id = Id(),
                 ipt_cols:Iterable,
                 out_cols:Iterable,
                 lamda_or_quantile:float = 1.,
                 lr:float = 1e-2,
                 order:int = 1, 
                 interaction_order:int = 0,
                 optimizer:str = 'adam', 
                 num_steps:int = int(1e3),
                 device:str = 'cpu',
                 **kwargs)->None|bool:

        self.order = order 
        self.interaction_order = interaction_order 
        self.num_epochs = int(num_steps)


        ipt_shape = len(ipt_cols)
        ipt_shape = 1 + ipt_shape*self.order 
        self.devide = 0
        out_shape = len(out_cols)
        
        
        self.polyI = ipt_shape**self.order
        self.I = 1 + self.polyI + ((interaction_order*(interaction_order-1)*ipt_shape*(ipt_shape-1))/4)
        
        self.model = LinearNN(self.I, out_shape, F)
        self.lamda_or_quantile = lamda_or_quantile
        self.device = device
        if(paramter_estimator not in options):
            raise ValueError(f'Only implemented for {str(options)}')
        elif(paramter_estimator in {'GLS', 'Ridge'}):
            self.Loss = torch.nn.MSELoss()
            self.fit_it = mapping[paramter_estimator]
            self.grad_ = False
            
        else:
            self.Loss = mapping[paramter_estimator](lam = self.lamda_or_quantile)
            self.grad_ = True
            if(isinstance(optimizer, str)):
                optimizer = opt_map[optimizer](lr=lr, **kwargs)
            self.Optim = opt_map[optimizer]
            self.fit_it = GradDescentTrain(self.model, self.Loss, self.Optim, epochs=self.num_epochs, device= device)
        return  
    
    def change_optim(self, opt:Optimizer, *args, **kwargs):
        self.Optim = opt(self.model.parameters, *args, **kwargs)
        return
    
    def mk_poly(self,x:Tensor)->Tensor:
        t = torch.empty((x.shape[0], self.I), dtype=x.dtype)
        t[:, 0] = 1
        t[1:1+self.polyI,:] = poly(x,self.order)
        if(self.interaction_order):
            t[1+self.polyI:, :] = poly_int(x, self.interaction_order, self.I-self.polyI-1)
        return t.to(self.device)
    
    def train(self, *args:tuple['x':torch.Tensor, 'y':torch.Tensor, 'xval':torch.Tensor, 'yval':torch.Tensor,]|tuple['trn':DataLoader, 'val':DataLoader], **kwargs:dict[Any])->None:
        self.model.to(self.device)
        if(self.grad_):
            self.fit_it(*args, **kwargs)
        else:
            self.model.parameters().data[0] = self.fit_it(args[0],args[1],self.lamda_or_quantile)
        self.model.requires_grad_(False).to('cpu')
        return
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)
@torch.jit.script
def poly(x:Tensor, order:int)->Tensor:
    X = torch.empty((x.shape[0], int(order**x.shape[1])))
    X[:,0] = 1.0
    for i in range(x.shape[1]):
        for o in range(order):
             X[:, (i*order)+o] = x[:, i]**o
    return X



@torch.jit.script
def poly_int(x:Tensor, O:int, sp:int):
    X = torch.empty((x.shape[0],  sp))
    D = x.shape[1]
    ct = 0
    for i in range(D - 1):
        for j in range(i + 1, D):
            for o in range(O):
                X[:,ct] = x[:, i * D + o] * x[:, j + D * o]
                ct += 1
    return X