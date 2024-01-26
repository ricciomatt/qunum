import numba as nb 
import numpy as np 

@nb.njit("UnicodeCharSeq(17)[:](int64[:])", parallel = True)
def nb_get_cols(arr:np.ndarray):
    r = np.empty((arr.shape[0]), dtype='U17')
    for i in nb.prange(arr.shape[0]):
        r[i] = "column_"+str(i)
    return r

def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)