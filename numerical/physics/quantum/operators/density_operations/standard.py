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
                pA[:, i,j] = p[:, ix[i], ix[j]].sum(dim = [1,2])
    return pA



@torch.jit.script
def ventropy(p:torch.Tensor)->float:
    Lam  = torch.linalg.eigvals(p)
    S = 0.0
    for i, lam in enumerate(Lam):
        S+=lam*torch.log(lam)
    return S

@torch.jit.script
def pT_arr(p, ixs):
    k = np.empty_like(p)
    for i in range(ixs.shape[0]):
        for j in range(ixs.shape[0]):
            t = [[t for m in range(ixs.shape[1])]
                 for t in ixs[i]]
            l = [ixs[j] for m in range(ixs[j].shape[0])]
            k[t,l] = (p[t,l].T)
    return k
    


