from scipy.sparse import csr_matrix
from .meta import OperMeta
from ..operators.density_operations import ptrace_np_ix as ptrace_ix, vgc
import polars as pl 
import numpy as np
from numpy.typing import NDArray

class OperatorCSR(csr_matrix):
    def __init__(self, 
                 x:NDArray, 
                 n_particles:int = None,
                 hilbert_space_dims:int = 2,
                 sparsify:bool = True,
                 meta:OperMeta|None = None):
        super(OperatorCSR, self).__init__()
        if(meta is None):
            self._metadata = OperMeta(n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, shp=self.shape)
        else:
            self._metadata = meta
        return
    
    def dag(self):
        return OperatorCSR(self, meta = self._metadata).conj().T
    
    def ptrace(self, keep_ix):
        a = vgc(keep_ix)
        ix_ =  np.array(
            self._metadata.ixs.groupby(
                pl.col(
                    a
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch().sort(a)['ix'].to_list()
            )[:,0]
        return  OperatorCSR(ptrace_ix(ix_, csr_matrix(self)), meta = self._metadata)
    