import polars as pl
import numpy as np
from sympy import Matrix, log, Function
import numba as nb
from numpy.typing import NDArray
def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)

@nb.jit(forceobj=True)
def ptrace_ix(ix:NDArray[np.int64], p:Matrix)->Matrix:
    pA = Matrix.zeros(ix.shape[0], ix.shape[0])
    for i in nb.prange(ix.shape[0]):
        for j in range(ix.shape[0]):
            pA[i,j] = p[ix[i], ix[j]].sum()
    return pA

@nb.jit(forceobj=True)
def ventropy(p:Matrix)->Function:
    ev = p.eigenvals()
    S = 0
    for lam in ev:
        for j in range(ev[lam]):
            if(lam != 0):
                S-=lam*log(lam)
    return S

@nb.jit(forceobj=True)
def pid(p:Matrix)->Matrix:
    return p 



def pT_arr(p, ixs):
    k = np.empty_like(p)
    for i in range(ixs.shape[0]):
        for j in range(ixs.shape[0]):
            t = [[t for m in range(ixs.shape[1])]
                 for t in ixs[i]]
            l = [ixs[j] for m in range(ixs[j].shape[0])]
            k[t,l] = (p[t,l].T)
    return k
    


