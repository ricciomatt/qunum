import numpy as np 
try:
    import cupy as cp 
    from cupy.typing import NDArray
except:
    import numpy as np
    from numpy.typing import NDArray
from typing import Callable
import torch
from .functions import Id

def lin_reg_do(X:NDArray,
               Y:NDArray, 
               inverse_function:Callable):
    if(isinstance(np.arange(100), np.ndarray)):
        XtXI = np.linalg.inv(X.T@ X)
    else:
        XtXI = torch.linalg.inv(X.T @ X)
    beta = XtXI @ (X.T @ inverse_function(Y))
    return beta

class GLR:
    def __init__(self,
                 order:int = 1,
                 forward_function:Callable = Id, 
                 inverse_function:Callable = Id,
                 cross_terms:bool = False
                 ):
        self.order = order
        self.forward_function = forward_function
        self.inverse_function = inverse_function
        return
    
    def mk_poly(self, x:torch.Tensor|np.ndarray):
        if(isinstance(x, np.ndarray)):
            t = np.ones((x.shape[0], 1+x.shape[1]*self.order), dtype=x.dtype)
        else:
            t = torch.ones((x.shape[0], 1+x.shape[1]*self.order), dtype=x.dtype)
        k = 1
        for i in range(1, self.order+1):
            for j in range(x.shape[1]):
                t[:,k] = x[:,j]**i
                k+=1
        return t
    
    def fit(self,x:torch.Tensor|np.ndarray, y:torch.Tensor|np.ndarray, mk_poly:bool = True):
        if(mk_poly):
            x = self.mk_poly(x)
        self.beta = lin_reg_do(x, y, self.inverse_function)
        self.compute_errs(x, y, yh=self.__call__(x, mk_poly), mk_poly=False)
        return 
    
    def compute_errs(self, x:torch.Tensor|np.ndarray, y:torch.Tensor|np.ndarray, yh:torch.Tensor|np.ndarray|None = None, mk_poly:bool = True):
        if(yh is None):
            yh = self.__call__(x,mk_poly)
        return
    
    def __call__(self, x:torch.Tensor|np.ndarray, mk_poly:bool = True):
        if(mk_poly):
            x = self.mk_poly(x)
        return self.forward_function(torch.einsum('Ai, ij-> Aj',x, self.beta))
    
    def __repr__(self):
        return f''''''
    
