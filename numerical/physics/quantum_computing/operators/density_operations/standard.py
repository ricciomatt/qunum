import polars as pl
import numpy as np
import numba as nb
from numpy.typing import NDArray
import torch

def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)

@nb.jit(parallel = True, fastmath = True)
def ptrace_np_ix(ix:NDArray[np.int64], p:NDArray)->NDArray:
    pA = NDArray.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype)
    for i in nb.prange(ix.shape[0]):
        for j in nb.prange(ix.shape[0]):
            pA[i,j] = p[ix[i], ix[j]].sum()
    return pA

@torch.jit.script
def ptrace_torch_ix(ix:torch.Tensor, p:torch.Tensor)->torch.Tensor:
    pA = torch.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype)
    for i in range(ix.shape[0]):
        for j in range(ix.shape[0]):
            pA[i,j] = p[ix[i], ix[j]].sum()
    return pA

