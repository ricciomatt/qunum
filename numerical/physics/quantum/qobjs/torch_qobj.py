import torch
import polars as pl
from numpy.typing import NDArray
from .meta import QobjMeta
from warnings import warn
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
from ..operators.density_operations import ptrace_torch_ix as ptrace_ix, vgc, pT_arr, ventropy
import torch
import polars as pl
from numpy.typing import NDArray
from typing import Sequence
from warnings import warn
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor

class TQobj(Tensor):
    def __new__(cls, 
                data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 sparsify:bool = True,
                 **kwargs):
        #obj = super(TQobj,cls).__new__(cls, data,*args, dtype = torch.complex64,**kwargs)
        if(isinstance(data, torch.Tensor)):
            data = torch.tensor(data.detach().numpy(), dtype=torch.complex64)
        else:
            data = torch.tensor(data, dtype=torch.complex64)
        obj = super(TQobj, cls).__new__(cls, data, *args, **kwargs)
        return obj
    def __init__(self, 
                 data,
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
        
        #if(sparsify):
         #   self.to_sparse_qobj()
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
        if(len(self.shape)==3):
            
            return TQobj(torch.transpose(self.data.resolve_conj(), 1,2), meta= meta)
        else: 
            return TQobj(self.data.resolve_conj().T.numpy(), meta= meta)

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
        if not (isinstance(O, TQobj) or isinstance(O,Tensor)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__mul__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return TQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
   
    def __rmatmul__(self, O:object|Tensor)->object:
        return self.__matmul__(O)
    
    
    def __mul__(self, O:object|Tensor)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__mul__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return TQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
    
    def __rmul__(self, O:object|Tensor)->object:
        return self.__mul__(O)
    
    
    def __add__(self, O:object|Tensor|float|int)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor) or isinstance(O,float) or isinstance(O,int) or isinstance(O,complex)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__add__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return TQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
    
    def __radd__(self, O:object|Tensor)->object:
        return self.__add__(O)
    
    def entropy(self,)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        return ventropy(torch.tensor(self.data.detach().numpy(), dtype = torch.complex64))
    
    def __getitem__(self, index):
        item = super(TQobj, self).__getitem__(index)
        try:
            item._metadata = self._metadata
        except:
            pass
        return item
