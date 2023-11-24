from sympy import Matrix, eye
import numpy as np
import polars as pl
from numpy.typing import NDArray
from .meta import SQObjMeta
from ..operations import ptrace_ix, vgc, ventropy
from warnings import warn


class SQObj(Matrix):
    def __init__(self, 
                 *args,
                 meta:SQObjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 check_hermitian:bool = False,
                 **kwargs)->object:
        super(SQObj, self).__init__()
        if(meta is None):
            self._metadata = SQObjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape,
                check_hermitian = check_hermitian
             )
        else:
            self._metadata = meta
        if(self._metadata.check_hermitian):
            if(self._metadata.obj_tp == 'operator'):
                self._metadata.herm = bool(eye(self.shape[0], self.shape[1])  == self.dag() @ self)
            else:
                warn('Cannot Check Hermicity of Object type: '+self._metadata.obj_tp)
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
    
    
    
