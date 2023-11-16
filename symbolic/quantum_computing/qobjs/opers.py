from sympy import Matrix, eye, sqrt
import numpy as np
import torch
import numba as nb
import warnings 
import polars as pl
from numpy.typing import NDArray
from .meta import OperMeta
from ..operations import ptrace_ix, vgc

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
    
