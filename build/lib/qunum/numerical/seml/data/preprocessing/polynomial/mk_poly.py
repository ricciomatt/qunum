import torch 
from torch import Tensor
from typing import Sequence,Iterable, Any 
import numpy as np 
from itertools import combinations
from copy import deepcopy
class ComputePoly:
    def __init__(self, ipt_cols:Sequence[int]=  None, order:int = 1, interaction_order:int = 0 , interaction_cols:Sequence[int]|None = None, **kwargs:dict[Any]):
        self.order = order
        self.interaction_order = interaction_order
        ipt_shp = len(ipt_cols)
        if(interaction_order == 0 and interaction_cols is None):
            self.int_dims = []
        else:
            self.int_dims = list(map(lambda icol: np.where(ipt_cols == icol)[0][0], interaction_cols))
        self.order = order 
        self.interaction_order = interaction_order 
        self.polyI = ipt_shp**self.order
        self.I = 1 + self.polyI + ((interaction_order*(interaction_order-1)*len(interaction_cols)*(len(interaction_cols)-1))/4)
        return
    
    def __call__(self, x:Tensor)->Tensor:
        return mk_poly(x,self.I, self.polyI, self.order, self.interaction_order, self.int_dims)

@torch.jit.script
def poly(x:Tensor, order:int, polyI:int)->Tensor:
    X = torch.empty((x.shape[0], int(polyI)))
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

@torch.jit.script
def mk_poly(x:Tensor, I:int, polyI:int, order:int, interaction_order:int, int_dims:list[int])->Tensor:
    ret_tensor = torch.empty((x.shape[0], I), dtype=x.dtype)
    ret_tensor[:, 0] = 1.0
    ret_tensor[:, 1:1+polyI] = poly(x, order, polyI)
    if(I != polyI+1):
        ret_tensor[:, 1+polyI:] = poly_int(x[:, int_dims], interaction_order, I - polyI - 1)
    return ret_tensor







