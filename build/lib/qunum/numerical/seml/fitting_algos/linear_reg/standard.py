import numpy as np 
import torch
from numpy.typing import NDArray
from typing import Callable
from torch import Tensor
from .out_functions import *

@torch.jit.script
def reg(X:Tensor, Y:Tensor)->tuple[Tensor, Tensor]:
    XtXi = torch.linalg.inv(X.T @ X)
    return XtXi @ X.T @ Y, XtXi

def lin_reg_do(X:NDArray|Tensor,
               Y:NDArray|Tensor, 
               F:Callable)->Tensor:
    return reg(X, F.inv(Y))

class GLR:
    def __init__(self,
                 order:int = 1,
                 F:Callable = Id(use_numpy=False), 
                 cross_terms:bool = False,
                 device:int|str = 'cpu',
                 ):
        self.order = order
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
    
    def fit(self,x:torch.Tensor|np.ndarray, y:torch.Tensor|np.ndarray, mk_poly:bool = True):
        if(mk_poly):
            x = self.mk_poly(x)
        self.beta, self.XtXi = lin_reg_do(x, y, self.F)
        self.compute_errs(x, y, yh=self.__call__(x, mk_poly), mk_poly=False)
        return 
    
    def compute_stats(self, xtrn:torch.Tensor|np.ndarray, ytrn:torch.Tensor|np.ndarray, xtst:torch.Tensor|np.ndarray, ytst:torch.Tensor|np.ndarray, mk_poly:bool=True)->dict:
        return
    
    def __call__(self, x:torch.Tensor|np.ndarray, mk_poly:bool = True):
        if(mk_poly):
            x = self.mk_poly(x)
        return self.F(torch.einsum('Ai, ij-> Aj',x, self.beta))
    
    def __repr__(self):
        return f''''''
    
