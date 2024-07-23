import polars as pl
import numpy as np
import numba as nb
from numpy.typing import NDArray
import torch
from torch import Tensor, kron
from ....meta import QobjMeta

@nb.njit("UnicodeCharSeq(17)[:](int64[:])", parallel = True)
def nb_get_cols(arr:np.ndarray):
    r = np.empty((arr.shape[0]), dtype='U17')
    for i in nb.prange(arr.shape[0]):
        r[i] = "column_"+str(i)
    return r

def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)
'''
@nb.njit('complex128[:,:,:](int64[:,:], complex128[:,:,:])', parallel = True, fastmath = True)
def nb_ptrace_ix(ix:NDArray[np.int64], p:NDArray)->NDArray:
    pA = np.empty((p.shape[0],ix.shape[0], ix.shape[0]), dtype = p.dtype)
    for i in nb.prange(ix.shape[0]):
        for j in nb.prange(ix.shape[0]):
            pA[:, i, j] = p[:, ix[i], ix[j]].sum()
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
def ptrace_bwd_torch_ix(ix:torch.Tensor, p:TQobj)->TQobj:
    if(len(p.shape) == 2):
        pA = TQobj(torch.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype, dtype = p.dtype, requires_grad=p.requires_grad))
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] = p[ix[i], ix[j]].sum()
    else:
        pA = TQobj(torch.zeros(p.shape[0], ix.shape[0], ix.shape[0], dtype = p.dtype ), requires_grad=p.requires_grad,  hilbert_space_dims=p._metadata.hilbert_space_dims)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA


@torch.jit.script
def ventropy(p:torch.Tensor, epsi:float = 1e-8)->torch.Tensor:
    if(len(p.shape) == 2):
        Lam  = torch.linalg.eigvals(p).real
        S = torch.tensor([0.0])
        logLam = torch.log(Lam)
        ix = torch.where(torch.isnan(logLam) | torch.isinf(logLam))[0]
        logLam[ix] = 0
        S-=(Lam*logLam).sum()
    else:
        Lam  = torch.linalg.eigvals(p).real
        S = torch.zeros_like(Lam[:,0])
        for i in range(Lam.shape[1]):
            logLam = torch.log(Lam[:,i])
            ix = torch.where(torch.isnan(logLam) | torch.isinf(logLam))[0]
            logLam[ix] = 0
            S-=Lam[:,i]*logLam   
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
    

@torch.jit.script
def ptrace_loc_ix(ix:Tensor, p:TQobj)->TQobj:
    if(len(p.shape) == 2):
        pA = torch.zeros((ix.shape[0], ix.shape[0]), dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] = p[ix[i], ix[j]].sum()
    else:
        pA = torch.zeros((p.shape[0], ix.shape[0], ix.shape[0]), dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA



from ..torch_qobj import TQobj