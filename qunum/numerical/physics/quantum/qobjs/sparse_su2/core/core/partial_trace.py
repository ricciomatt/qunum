from torch import Tensor, jit, from_numpy, ceil, unique, vmap, arange as tarange, einsum
from numpy import arange, delete, ndarray

def ptrace(aHat:Tensor, aC:Tensor, keep_ix:Tensor, TrOut:Tensor)->Tensor:  
    rC = set_up_trace(aHat[:, TrOut], aC)
    rHat, idx = unique(aHat[:, keep_ix], dim = 0, return_inverse = True)
    return rHat, einsum('Ai, A->i', vmap(lambda x: (x==idx).to(rC.dtype), out_dims=1)(tarange(rHat.shape[0])), rC)

def fullTrace(aHat:Tensor, aC:Tensor)->Tensor:
    return set_up_trace(aHat,aC).sum()

@jit.script
def set_up_trace(eHat:Tensor, c:Tensor)->Tensor:
    b = (ceil(eHat[...,0]) == 1).all(dim=1).to(eHat.dtype)
    return (b * (c * eHat[...,0].sum(dim=1)))