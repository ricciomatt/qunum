from sympy import Matrix, eye, sqrt
import numpy as np
import torch
import numba as nb
import warnings 
import polars as pl
from numpy.typing import NDArray

class OperMeta:
    def __init__(self, n_particles:int= None, hilbert_space_dims:int = 2, shp:tuple[int] = None)->None:
        if(hilbert_space_dims**n_particles == shp[0]):
            self.n_particles = n_particles
            self.hilber_space_dims = hilbert_space_dims
            self.ixs = pl.DataFrame(
                np.array(
                    np.meshgrid(
                        *[np.arange(hilbert_space_dims)]*n_particles)
                    ).T.reshape(
                    -1, 
                    n_particles
                    )).with_row_count().lazy()
        elif(hilbert_space_dims == 2):
            self.hilber_space_dims = hilbert_space_dims
            self.n_particles = n_particles
            self.ixs = pl.DataFrame(
                np.array(
                    np.meshgrid(
                        *[np.arange(hilbert_space_dims)]*n_particles)
                    ).T.reshape(
                    -1, 
                    n_particles
                    )).with_row_count().lazy()
            warnings.warn('Assuming that this is a 2d hilbert space')
        else:
            raise RuntimeError('Operators must have dimensions specified')
        return
    
class Operator(Matrix):
    def __init__(self, *args, 
                 meta:OperMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 **kwargs)->object:
        super(Operator, self).__init__()
        if(meta is None):
            self._metadata = OperMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        return
    
    def dag(self)->object:
        return Operator(self.conjugate().T, self._metadata)

    def ptrace(self, keep_ix:tuple[int]|list[int])->object:
        ix_ =  np.array(
            self._metadata.ixs.groupby(
                pl.col(
                    vgc(keep_ix))
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch()['ix'].to_list()
            )[:,0]
        return Operator(ptrace_ix(ix_, np.array(self)), meta = self._metadata)
    
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
