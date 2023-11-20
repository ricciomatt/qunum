import torch
from .meta import OperMeta
from ..operators.density_operations import ptrace_torch_ix as ptrace_ix, vgc
import polars as pl 
import numpy as np

class OperatorTorch(torch.Tensor):
    def __init__(self, 
                 x:torch.Tensor, 
                 n_particles:int = None,
                 hilbert_space_dims:int = 2,
                 sparsify:bool = True,
                 meta:OperMeta|None = None):
        super(OperatorTorch, self).__init__()
        if(meta is None):
            self._metadata = OperMeta(n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, shp=self.shape)
        else:
            self._metadata = meta
        if(sparsify):
            self.to_sparse()
        return
    
    def dag(self):
        return OperatorTorch(self, meta = self._metadata).conj().T
    
    def ptrace(self, keep_ix):
        a = vgc(keep_ix)
        ix_ =  torch.tensor(np.array(
            self._metadata.ixs.groupby(
                pl.col(
                    a
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch().sort(a)['ix'].to_list()
            )[:,0])
        return  OperatorTorch(ptrace_ix(ix_, torch.tensor(self)), meta = self._metadata)
    def pT(self):
        pass
    
class StateTorch(torch.Tensor):
    def __init__(self, 
                 x:torch.Tensor, 
                 n_particles:int,
                 hilbert_space_dims:int,
                 sparsify:bool = True):
        super(OperatorTorch, self).__init__()
        self._metadata = OperMeta(n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, shp=self.shape)
        pass