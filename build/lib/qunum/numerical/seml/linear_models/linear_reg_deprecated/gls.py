import numpy as np 
import torch
from numpy.typing import NDArray
from typing import Callable, Any
from torch import Tensor
from .out_functions import *
from sklearn.linear_model import Lasso

@torch.jit.script
def reg(X:Tensor, Y:Tensor)->tuple[Tensor, Tensor]:
    XtXi = torch.linalg.inv(X.T @ X)
    return XtXi @ X.T @ Y, XtXi

class GLS:
    def __init__(self,
                 order:int = 1,
                 F:Callable|str = Id(use_numpy=False), 
                 cross_terms:bool = False,
                 device:int|str = 'cpu',
                 **kwargs,
                 ):
        self.order = order
        if(isinstance(F, str)):
            F = getMap()[F](**kwargs)
        self.F = F
        self.cross_terms = cross_terms
        self.device = device
        self.stats = None
        return
    def mk_poly(self, x:torch.Tensor|np.ndarray):
        if(isinstance(x, np.ndarray)):
            x = torch.tensor(x)
            t = torch.ones((x.shape[0], 1+x.shape[1]*self.order), dtype=x.dtype)
        else:
            t = torch.ones((x.shape[0], 1+x.shape[1]*self.order), dtype=x.dtype)
        k = 1
        for i in range(1, self.order+1):
            for j in range(x.shape[1]):
                t[:,k] = x[:,j]**i
                k+=1
        return t.to(self.device)
    
    def train(self,
              x:torch.Tensor|np.ndarray, y:torch.Tensor|np.ndarray, 
              xtst:torch.Tensor|np.ndarray, ytst:torch.Tensor|np.ndarray,

              *args:tuple[torch.Tensor|np.ndarray, torch.Tensor|np.ndarray, Any],  
              mk_poly:bool = True, **kwargs:dict[Any]):
        if(mk_poly):
            x = self.mk_poly(x)
        self.beta, self.XtXi = reg(x, self.F.inv(y))
        return self.compute_stats(x, y, xtst, ytst, mk_poly=False)
    
    def compute_stats(self, xtrn:torch.Tensor|np.ndarray, ytrn:torch.Tensor|np.ndarray, xtst:torch.Tensor|np.ndarray, ytst:torch.Tensor|np.ndarray, mk_poly:bool=True)->dict:
        yh = self(xtrn, mk_poly = False)
        yhtst = self(xtst)
        df = {}
        return 

    def compute_errs(self, y, yh):
        res = y- yh
        Res_2 = torch.einsum('Ai, Ai->i', res, res)
        
        #.. 
        #sigma_beta = 
        #chi2 = 
        self.stats = None 
        return
    
    def __call__(self, x:torch.Tensor|np.ndarray, mk_poly:bool = True):
        if(mk_poly):
            x = self.mk_poly(x)
        return self.F(torch.einsum('Ai, ij-> Aj', x, self.beta))
    
    def __repr__(self):
        return f''''''
    
