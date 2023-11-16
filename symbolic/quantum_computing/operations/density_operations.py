import polars as pl
import numpy as np
from sympy import Matrix
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
