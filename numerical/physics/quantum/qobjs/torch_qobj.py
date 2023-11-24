import torch
import polars as pl
from numpy.typing import NDArray
from .meta import QobjMeta
from warnings import warn
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
from ..operators.density_operations import ptrace_torch_ix as ptrace_ix, vgc, pT_arr, ventropy
class TQobj(Tensor):
    def __init__(self, 
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 sparsify:bool = True,
                 **kwargs)->object:
        super(TQobj, self).__init__()
        if(meta is None):
            self._metadata = QobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        
        if(sparsify):
            self.to_sparse_qobj()
        return
    
    def to_sparse_qobj(self)->None:
        self = self.to_sparse_csr()
        return 
    
    def dag(self)->object:
        meta = copy.copy(self._metadata)
        if(self._metadata.obj_tp == 'ket'):
            meta.obj_tp = 'bra'
        elif(self._metadata.obj_tp == 'bra'):
            meta.obj_tp = 'ket'
        return TQobj(self.conjugate().T, meta= meta)

    def ptrace(self, keep_ix:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        a = vgc(keep_ix)
        ix_ =  torch.tensor(
            self._metadata.ixs.groupby(
                pl.col(
                    a
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch().sort(a)['ix'].to_list()
            )[:,0]
        return TQobj(ptrace_ix(ix_, torch.tensor(self)), meta = self._metadata)
    
    def pT(self, ix_T:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        a = vgc(ix_T)
        ix_ =  torch.tensor(
            self._metadata.ixs.groupby(
                pl.col(
                    a
                    )
                ).agg(
                    pl.col('row_nr').implode().alias('ix')
                ).fetch().sort(a)['ix'].to_list()
            )[:,0]
        return TQobj(pT_arr(torch.tensor(self), ix_), meta = self._metadata)
    
    def __matmul__(self, O:object|Tensor)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = self.__mul__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return TQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilber_space_dims)
    
    def entropy(self, ix:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        return (ventropy(torch.tensor(self)), self._metadata)
    
    def __repr__(self):
        try:
            disp(md('Object Type: '+self._metadata.obj_tp))
            disp(md('Particles: '+ str(self._metadata.n_particles)+', Hilbert: '+str(self._metadata.hilber_space_dims)))
        except:
            pass
        return self.__str__()
