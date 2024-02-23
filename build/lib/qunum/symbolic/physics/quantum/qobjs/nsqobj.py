from sympy import Matrix, Symbol, kronecker_product as kron
from numpy import ndarray
import numpy as np 
from typing import Iterable
from .....numerical.physics.quantum.qobjs.meta import QobjMeta
from copy import copy, deepcopy
class SymbQobj(Matrix):
    def __init__(self,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int= 2,
                 dims:None|dict[str:int] = None,
                 **kwargs):
        super(SymbQobj, self).__init__()
        self.set_meta(meta = meta, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dims = dims, )
        pass

    def set_meta(self,
                 meta:QobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 dims:None|dict[int:set] = None ,):
        if(meta is None):
            self._metadata = QobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                dims = dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        return 
    
   
    def __matmul__(self, O:object|Matrix)->object:
        if not (isinstance(O, SymbQobj) or isinstance(O, Matrix)):
            raise TypeError('Must Be SymbQobj or Matrix')
        M = super(SymbQobj, self).__matmul__(O)
        M.set_meta(meta = QobjMeta(n_particles=self._metadata.n_particles, hilbert_space_dims=self._metadata.hilbert_space_dims, dims = self._metadata.dims, shp=M.shape))
        return M
    
    def __mul__(self, O:object|float|int|Symbol|Matrix)->object:
        M = super(SymbQobj, self).__mul__(O)
        M.set_meta(meta = self._metadata)
        return M
    
    def __add__(self, O:object|Symbol|float|int|Matrix)->object:
        M = super(SymbQobj, self).__add__(O)
        M.set_meta(meta =self._metadata)
        return M
    
    def __sub__(self, O:object|Symbol|float|int|Matrix)->object:
        M = super(SymbQobj, self).__sub__(O)
        M.set_meta(meta= self._metadata)
        return M
    
    def __radd__(self, O:object|Symbol|float|int|Matrix)->object:
        return self.__add__(O)
    
    def __rsub__(self, O:object|Symbol|float|int|Matrix)->object:
        return self.__sub__(O)
    
    def __rmatmul__(self, O:object|Symbol|float|int|Matrix)->object:
        return self.__matmul__(O)
    
    def __rmul__(self, O:object|Symbol|float|int|Matrix)->object:
        return self.__mul__(O)
    
    def __truediv__(self, O:object|Symbol|float|int|Matrix)->object:
        M = super(SymbQobj, self).__truediv__(O)
        M.set_meta(meta= self._metadata)
        return M

    
    def __repr__(self)->str:
        try:
            str_ = self._metadata.__str__()
        except:
            str_ = 'No dataset available'
        return str_+'\n'+super(SymbQobj, self).__repr__()
    def __str__(self)->str:
        return self.__repr__()
    
    def __xor__(self, O:object)->object:
        return direct_prod(self,O)
    
    def __rxor__(self, O:object) -> object:
        return direct_prod(O,self)

    def dag(self)->object:
        meta = copy(self._metadata)
        if(self._metadata.obj_tp == 'ket'):
            meta.obj_tp = 'bra'
        elif(self._metadata.obj_tp == 'bra'):
            meta.obj_tp = 'ket'
        return SymbQobj(self.adjoint(), meta= meta)
    def to_density(self)->object:
        pass

    def Tr(self, tr_out:list[int]|tuple[int]|ndarray|slice|int|Iterable|None=None, keep:list[int]|tuple[int]|ndarray|slice|int|Iterable|None=None, reorder:bool = False)->object:
        if(tr_out is None and keep is None):
            return self.trace()
        else:
            if(tr_out is not None):
                tr_out = self._metadata.check_particle_ixs(tr_out)
                ix = np.arange(self._metadata.n_particles)
                ix = np.delete(ix, tr_out)
            else:
                ix = self._metadata.check_particle_ixs(keep)
            if(ix.shape[0] == self._metadata.n_particles):
                return self
            ix_ =  (self._metadata.query_particle_ixs(ix))[:,0]
            meta = copy(self._metadata)
            meta.update_dims(ix, reorder=reorder)
            pA = SymbQobj(Matrix().zeros(ix_.shape[0], ix_.shape[0]), meta = meta)
            return ptrace_bwd_ix(ix_, self, pA)
        
    def pT(self)->object:
        pass
    
    def sum(self)->object:
        return super(SymbQobj, self).sum(self) 
    
import numba as nb 
@nb.jit(forceobj=True)
def ptrace_bwd_ix(ix:ndarray, p:SymbQobj, pA:SymbQobj)->SymbQobj:
    for i in range(ix.shape[0]):
        for j in range(ix.shape[0]):
            pA[i,j] += p[ix[i], ix[j]].sum()
    return pA

'''
@torch.jit.script
def ptrace_bwd_ix(ix:Tensor, p:SymbQobj)->SymbQobj:
    if(len(p.shape) == 2):
        pA = torch.zeros((ix.shape[0], ix.shape[0]), dtype = p.dtype, requires_grad=p.requires_grad)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] = p[ix[i], ix[j]].sum()
    else:
        pA = torch.zeros((p.shape[0], ix.shape[0], ix.shape[0]), dtype = p.dtype, requires_grad=p.requires_grad)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA
'''

def direct_prod(*args:tuple[SymbQobj])->SymbQobj:
    from string import ascii_uppercase
    A = args[0]
    assert isinstance(A, SymbQobj), f'Arguments must be Qunatum Objects(SymbQobj), item 0'
    dims = {ix:A._metadata.dims[d] for ix, d in enumerate(A._metadata.dims)}
    for i, a in enumerate(args[1:]):
        assert isinstance(a, SymbQobj), f'Arguments must be Qunatum Objects(SymbQobj), item {i}'
        assert isinstance(a._metadata, QobjMeta), f"Arguments must have valid Object Data, item {i}"     
        A = kron(A, a)
        tdims = {len(dims)+ix:a._metadata.dims[d] for ix, d in enumerate(a._metadata.dims)}
        dims.update(tdims)
    meta = QobjMeta(dims=dims, shp=A.shape)
    return SymbQobj(A, meta = meta)

