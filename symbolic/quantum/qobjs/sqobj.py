from sympy import Matrix
import numpy as np
import polars as pl
from numpy.typing import NDArray
from .meta import SQObjMeta
from ..operations import ptrace_ix, vgc, ventropy


class SQObj(Matrix):
    def __init__(self, 
                 *args,
                 meta:SQObjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 **kwargs)->object:
        super(SQObj, self).__init__()
        if(meta is None):
            self._metadata = SQObjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        return
    
    def dag(self)->object:
        return SQObj(self.conjugate().T, self._metadata)

    def ptrace(self, keep_ix:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
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
        return SQObj(ptrace_ix(ix_, np.array(self)), meta = self._metadata)
    
    def pT(self, ix_T:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        a = vgc(ix_T)
        ix_ =  np.array(
            self._metadata.ixs.groupby(
                pl.col(
                    a
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch().sort(a)['ix'].to_list()
            )[:,0]
        return
    
    def entropy(self, ix:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        return (ventropy(Matrix(self)), self._metadata)
    
    
    
