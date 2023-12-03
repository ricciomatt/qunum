import polars as pl
import numpy as np
import numba as nb
from numpy.typing import NDArray
import torch

def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)
'''
@nb.jit(parallel = True, fastmath = True)
def ptrace_np_ix(ix:NDArray[np.int64], p:NDArray)->NDArray:
    pA = NDArray.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype)
    for i in nb.prange(ix.shape[0]):
        for j in nb.prange(ix.shape[0]):
            pA[i,j] = p[ix[i], ix[j]].sum()
    return pA
'''

@torch.jit.script
def ptrace_torch_ix(ix:torch.Tensor, p:torch.Tensor)->torch.Tensor:
    if(len(p.shape) == 2):
        pA = torch.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] = p[ix[i], ix[j]].sum()
    else:
        pA = torch.zeros(p.shape[0], ix.shape[0], ix.shape[0], dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] = p[:, ix[i], ix[j]].sum(dim = [1])
    return pA



@torch.jit.script
def ventropy(p:torch.Tensor, epsi:float = 1e-8)->torch.Tensor:
    if(len(p.shape) == 2):
        Lam  = torch.linalg.eigvals(p)
        S = torch.tensor([0.0])
        for i, lam in enumerate(Lam):
            t = lam*torch.log(lam+epsi)
            S+=t      
    else:
        Lam  = torch.linalg.eigvals(p)
        S = torch.zeros(p.shape[0], dtype=Lam.dtype)
        for i in range(Lam.shape[1]):
            S+=Lam[:,i]*torch.log(Lam[:,i]+epsi)
    return S

@torch.jit.script
def pT_arr(p:torch.Tensor, ixs:torch.Tensor):
    k = torch.empty_like(p)
    for i in range(ixs.shape[0]):
        for j in range(ixs.shape[0]):
            t = [[t for m in range(ixs.shape[1])]
                 for t in ixs[i]]
            l = [ixs[j] for m in range(ixs[j].shape[0])]
            k[t,l] = (p[t,l].T)
    return k
    


