from sympy import Matrix, eye, MatAdd, Symbol
import numpy as np
import polars as pl
from numpy.typing import NDArray
from .meta import SQobjMeta
from ..operations import ptrace_ix, vgc, ventropy
from warnings import warn
from ..operations.density_operations.common_opers import pT_arr
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from sympy import kronecker_product as dprod



class SQobj(Matrix):
    def __init__(self, 
                 *args,
                 meta:SQobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 **kwargs)->object:
        super(SQobj, self).__init__()
        if(meta is None):
            self._metadata = SQobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        return
    
    def __matmul__(self, O:object|Matrix)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(SQobj,self).__mul__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return SQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
    
   
    
    def __mul__(self, O:object|Matrix|int|Symbol)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(SQobj,self).__mul__(O)
        if(M.shape == (1,1)):
            return M
        else:
            return SQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
    
    def __add__(self, O:object|Matrix)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(SQobj, self)._eval_add(O)
        if(M.shape == (1,1)):
            return M
        else:
            return SQobj(M, n_particles = meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims)
    
    def __rmatmul__(self, O:object|Matrix)->object:
        return self.__matmul__(O)
    
    def __rmul__(self, O:object|Matrix|int|Symbol)->object:
        return self.__mul__(O)
   
    def __radd__(self, O:object|Matrix)->object:
        return self.__add__(O)
    
    
    def dag(self)->object:
        meta = copy.copy(self._metadata)
        if(self._metadata.obj_tp == 'ket'):
            meta.obj_tp = 'bra'
        elif(self._metadata.obj_tp == 'bra'):
            meta.obj_tp = 'ket'
        return SQobj(self.adjoint(), meta= meta)

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
        return SQobj(ptrace_ix(ix_, np.array(self)), meta = self._metadata)
    
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
        return SQobj(pT_arr(np.array(self), ix_), meta = self._metadata)
    
        
    def entropy(self)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        return (ventropy(Matrix(self)))
    
    def __repr__(self):
        try:
            disp(md('Object Type: '+self._metadata.obj_tp))
            disp(md('Particles: '+ str(self._metadata.n_particles)+', Hilbert: '+str(self._metadata.hilbert_space_dims)))
        except:
            pass
        return self.__str__()
    
    def __xor__(self, O:object)->object:
        return direct_prod(self, O)
    
    def __rxor__(self, O:object)->object:
        return direct_prod(O,self)
    
def direct_prod(A:SQobj, B:SQobj):
    if(not isinstance(A,SQobj) or not isinstance(B,SQobj)):
        raise TypeError('Must Be SQobj type')
    if(A._metadata.hilbert_space_dims == B._metadata.hilbert_space_dims):
        return SQobj(dprod(A,B), n_particles=A._metadata.n_particles+B._metadata.n_particles, 
                     hilbert_space_dims=A._metadata.hilbert_space_dims)
    else:
        raise ValueError('Hilbert Space Dimensions must match')