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
from sympy import kronecker_product as kron
from torch import Tensor
from typing import Sequence, Iterable
import numba as nb
from itertools import combinations

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
    
    def Tr(self, tr_out:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, keep:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        if(tr_out is not None):
            ix = np.arange(self._metadata.n_particles)
            ix = np.delete(ix, tr_out)
            a = vgc(ix)
        else:
            if(isinstance(keep,int) ):
                keep = [keep]
            elif(isinstance(keep, slice)):
                ix = np.arange(self._metadata.n_particles)[keep]
                keep = ix.copy()
            a = vgc(keep)
        ix_ =  np.array(
                self._metadata.ixs.groupby(
                    pl.col(
                        a
                        )
                    ).agg(
                        pl.col('row_nr').implode().alias('ix')
                    ).fetch().sort(a)['ix'].to_list()
                )[:,0]
        meta = copy.copy(self._metadata)
        meta.n_particles -= 1
        return SQobj(ptrace_ix(ix_, np.array(self)), meta = self._metadata)
    
    def pidMatrix(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, Projs:list[object])->NDArray:
        if(self._metadata.obj_tp is not 'operator'):
            raise TypeError('Not implimented for bra and kets')
        try:
            iter(A)
            A = list(A)
        except:
            A = [A]
        B = self.get_systems(A)
        I_aB = np.empty(self._metadata.hilbert_space_dims*len(A), B.shape[0], dtype=np.object_)
        for i,P in enumerate(Projs):
            rhoa = P @ self @ P.dag()
            rhoa/= rhoa.Tr()
            Ha = (rhoa.Tr(keep=A)).entropy()
            for j, b  in enumerate(B):
                k = copy.deepcopy(A)
                k.extend(b)
                I_aB[i, j] = Ha + rhoa.Tr(keep=list(b)).entropy() - rhoa.Tr(keep = k).entropy()
        return I_aB, B
    
    
    def get_systems(self, A):
        combs = []
        ix = np.arange(self._metadata.n_particles)
        ix = np.delete(ix, A)
        for i in range(ix.shape[0]+1):
            combs.extend(combinations(ix,i))
        return np.array(combs)
    
    
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
    
    def mutual_info(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, B:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        try:
            iter(A)
            A = list(A)
        except:
            A = [A]
        try:
            iter(B)
            B = list(B)
        except:
            B = [B]
        ix = copy.deepcopy(A)
        ix.extend(B)
        rhoA = self.Tr(keep = A)
        rhoB = self.Tr(keep = B)
        rhoAB = self.Tr(keep = ix)
        return rhoA.entropy() + rhoB.entropy() - rhoAB.entropy()
    
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
    
def direct_prodd(A:SQobj, B:SQobj):
    if(not isinstance(A,SQobj) or not isinstance(B,SQobj)):
        raise TypeError('Must Be SQobj type')
    if(A._metadata.hilbert_space_dims == B._metadata.hilbert_space_dims):
        return SQobj(kron(A,B), n_particles=A._metadata.n_particles+B._metadata.n_particles, 
                     hilbert_space_dims=A._metadata.hilbert_space_dims)
    else:
        raise ValueError('Hilbert Space Dimensions must match')
    
#@nb.jit(forceobj = True)
def direct_prod(*args:tuple[SQobj])->SQobj:
    A = args[0]
    if(not isinstance(A, SQobj)):
        A = A[0]
        args = args[0]
        if(not isinstance(A, SQobj)):
            raise TypeError('Must be SQobj')
    
    m = A._metadata.n_particles
    h = A._metadata.hilbert_space_dims
    print(A)
    for i, a in enumerate(args[1:]):
        if(isinstance(a, SQobj)):
            try:
                A = kron(A ,a)
                m+=a._metadata.n_particles
            except:
                ValueError('Must Have Particle Number')
        else:
            raise TypeError('Must be TQobj')
    meta = SQobjMeta(n_particles=m, hilbert_space_dims=h, shp=A.shape)
    return SQobj(A, n_particles=m, hilbert_space_dims=h, meta = meta)