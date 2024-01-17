import torch
import polars as pl
from numpy.typing import NDArray
from .meta import QobjMeta
from typing import Any, Iterable
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
from .density_operations import ptrace_torch_ix as ptrace_ix, vgc, nb_get_cols, pT_arr, ventropy
from warnings import warn
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
import numpy as np 
from torch import kron
import numba as nb
from itertools import combinations, product
from ....algebra.representations import su
from ....algebra import commutator as comm
from scipy.linalg import logm

'''Need to Fix the Gradient Function'''
class TQobj(Tensor):
    def __new__(cls, 
                data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int|None = 2,
                 sparsify:bool = True,
                 dtype = torch.complex128,
                 **kwargs):
        #obj = super(TQobj,cls).__new__(cls, data,*args, dtype = torch.complex64,**kwargs)
        if('requires_grad' not in kwargs):
            try:
                kwargs['requires_grad'] = data.requires_grad
            except:
                kwargs['requires_grad'] = False

        if(isinstance(data, torch.Tensor)):
            data = torch.tensor(data.detach().numpy(), *args, dtype=data.dtype, **kwargs)
        else:
            data = torch.tensor(data, *args, dtype=dtype, **kwargs)
            
        obj = super(TQobj, cls).__new__(cls, data)
        return obj
    
    def __init__(self, 
                 data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int|None = None,
                 sparsify:bool = True,
                 dtype = torch.complex128,
                 **kwargs)->object:
        super(TQobj, self).__init__()
        self.set_meta(meta=meta, n_particles=n_particles,hilbert_space_dims=hilbert_space_dims,sparsify=sparsify)
       
        if(self.requires_grad):
            self.retain_grad()
        #if(sparsify):
         #   self.to_sparse_qobj()
        return
    
    def set_meta(self,
                 meta:QobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 sparsify:bool = True,):
        if(meta is None):
            self._metadata = QobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                shp = self.shape
             )
        else:
            self._metadata = meta
        
        return 
    
    def __matmul__(self, O:object|Tensor)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__matmul__(O)
        if(len(M.shape) == 0):
            pass
        else:
            meta = QobjMeta(n_particles=meta.n_particles, hilbert_space_dims=meta.hilbert_space_dims, shp=M.shape)
        M.set_meta(meta= meta)
        return M
    
    def __mul__(self, O:object|Tensor|float|int)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__mul__(O)
        M.set_meta(meta = meta)
        return M
    
    def __add__(self, O:object|Tensor|float|int)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor) or isinstance(O,float) or isinstance(O,int) or isinstance(O,complex)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__add__(O)
        M.set_meta(meta = meta)
        return M
    
    def __sub__(self, O:object|Tensor|float|int)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor) or isinstance(O,float) or isinstance(O,int) or isinstance(O,complex)):
            raise TypeError('Must Be TQobj or Tensor')
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__sub__(O)
        M.set_meta(meta= meta)
        return M
    
    def __radd__(self, O:object|Tensor)->object:
        return self.__add__(O)
    
    def __rsub__(self, O:object|Tensor)->object:
        return self.__sub__(O)
    
    def __rmatmul__(self, O:object|Tensor)->object:
        return self.__matmul__(O)
    
    def __rmul__(self, O:object|Tensor)->object:
        return self.__mul__(O)
    
    def __truediv__(self, O:object|Tensor)->object:
        try:
            meta = self._metadata
        except:
            meta = O._metadata
        M = super(TQobj, self).__div__(O)
        M.set_meta(meta= meta)
        return M

    
    def __repr__(self)->str:
        try:
            str_ = self._metadata.__str__()
        except:
            str_ = 'No meta data available'
        return str_+'\n'+super(TQobj, self).__repr__()
    def __str__(self)->str:
        return self.__repr__()
    
    def __xor__(self, O:object)->object:
        return direct_prod(self,O)
    
    def __rxor__(self, O:object) -> object:
        return direct_prod(O,self)

    def to_sparse_qobj(self)->None:
        self = self.to_sparse_csr()
        return 
    
    def zero_grad(self):
        try:
            self.grad.data.zero_()
        except:
            pass
        return 
    
    def get_grad(self):
        self.retain_grad()
        return self.grad
    
    def dag(self)->object:
        meta = copy.copy(self._metadata)
        if(self._metadata.obj_tp == 'ket'):
            meta.obj_tp = 'bra'
        elif(self._metadata.obj_tp == 'bra'):
            meta.obj_tp = 'ket'
        if(len(self.shape)==3):
            M = self.conj().swapaxes(1,2)
            M.set_meta(meta= meta)
            return M
        else: 
            M = self.conj().T
            M.set_meta(meta= meta)
            return M
    
    def expm(self)->object:
        meta = self._metadata
        M = torch.linalg.matrix_exp(self)
        M.set_meta(meta = meta)
        return M

    def logm(self)->object:
        return TQobj(logm(self.detach().numpy()), meta = self._metadata)
    
    def to_tensor(self)->Tensor:
        return torch.tensor(self.data.detach().numpy(), dtype= self.dtype)
    
    def Tr(self, tr_out:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, keep:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        if(tr_out is None and keep is None):
            return self.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        else:
            
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
            if(a.shape[0] == self._metadata.n_particles):
                return self
            ix_ =  torch.tensor(
                self._metadata.ixs.group_by(
                    pl.col(
                        a.tolist()
                        )
                    ).agg(
                        pl.col('row_nr').implode().alias('ix')
                    ).fetch().sort(a)['ix'].to_list()
                )[:,0]
            meta = copy.copy(self._metadata)
            meta.n_particles -= 1
        if(self.requires_grad):
            return ptrace_loc_ix(ix_, self)
        else:
            return TQobj(ptrace_ix(ix_, self.clone().detach()), meta = meta)  
    
    def pT(self, ix_T:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        a = vgc(np.array(ix_T, dtype = np.int64))
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
    
    def entropy(self,)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        return ventropy(self)
    
    def polarization_vector(self, particle:int)->torch.Tensor:
        if(self._metadata.obj_tp != 'operator'):
            raise ValueError('Must be an operator')
        if(self._metadata.hilbert_space_dims != 2):
            raise ValueError('Only 2^n, hilbert spaces supported')
        p = self.Tr(keep=particle)
        s = su.get_pauli(to_tensor= True)
        shp = self.shape
        if(len(shp)>3):
            raise TypeError('Only implemented for 2 and 3d opers')
        else:
            if(len(shp)>2):
                shp = (4,shp[0], 1)
            else:
                shp = (4,1,1)
        vec = torch.zeros(shp, dtype = self.dtype)
        for i in range(4):
            vec[i,:,0] += (s[i] @ p).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return vec[:,:,0]
    
    def mutual_info(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, B:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None = None)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        try:
            iter(A)
            A = list(A)
        except:
            A = [A]
        if(B is not None):
            try:
                iter(B)
                B = list(B)
            except:
                B = [B]

            ix = copy.deepcopy(A)
            ix.extend(B)
            rhoAB = self.Tr(keep = ix)
        else:
            rhoAB = self
        rhoA = self.Tr(keep = A)
        rhoB = self.Tr(keep = B)
        return rhoA.entropy() + rhoB.entropy() - rhoAB.entropy()
    
    def pauli_decomposition(self, ret_sig:bool = False, keep_all:bool = False)->tuple[dict|object]|tuple[dict]:
        if(self._metadata.hilbert_space_dims != 2):
            raise ValueError('To pauli decompose mus be 2d hilbert spaces ')
        sig = TQobj(su.get_pauli(to_tensor=True), n_particles=1)
        ix = product(*(range(4) for i in range(self._metadata.n_particles)))
        A = {}
        for i,x in enumerate(ix):
            t = direct_prod(*(sig[j] for j in x))
            a = (t @ self).Tr()
            if(keep_all):
                A[x] = a
            elif not (torch.all(a.real == 0) and torch.all(a.imag == 0)):
                A[x] = a
        R = [A]
        if(ret_sig):
            R.append(sig)
        return tuple(R)
    
    def block_decimate(self)->dict[dict[str:Tensor]]:
        P, sig = self.pauli_decomposition(ret_sig = True, keep_all= False)
        Blocks = {}
        keys = list(P.keys())
        k = 0
        while len(P) != 0:
            p0 = keys[0]
            A = {p0: P[p0]}
            t = direct_prod(*(sig[n] for n in p0))
            j = k+1
            while j<len(keys):
                p1 = keys[j]
                b = direct_prod(*(sig[n] for n in p1))
                c = comm(t,b)
                if torch.all(c.real == 0) and torch.all(c.imag == 0):
                    A[p1] = P[p1]
                    del P[p1]
                    del keys[j]
                else:
                    j+=1
            del P[p0]
            del keys[0]
            Blocks[k] = A
            k+=1
        return Blocks
    
    def to_density(self):
        if(self._metadata.obj_tp  == 'operator'):
            raise TypeError('Must be ket or bra vector')
        elif(self._metadata.obj_tp == 'ket'):
            return self @ self.dag()
        else:
            return self.dag() @ self
    
    def pidMatrix(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, Projs:object)->Tensor:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Not implimented for bra and kets')
        try:
            iter(A)
            A = list(A)
        except:
            A = [A]
        rhoa = TQobj(Projs @ self @ Projs.dag())
        rhoa/= rhoa.Tr()
        rhoa = TQobj(rhoa)
        Ha = (rhoa.Tr(keep=A)).entropy()
        B = self.get_systems(A)
        I_aB = torch.empty(Ha.shape[0],B.shape[0])
        for i,b in enumerate(B):
            I_aB[:, i] = rhoa.mutual_info(A,b)
        return I_aB, B
    
    def cummatprod(self):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must Be operator type')
        if(len(self.shape) != 3):
            raise TypeError('Must Be operator type with multiple entries rank(3)')
        return TQobj(cummatprod_(self.to_tensor()), meta = self._metadata)
        
    def get_systems(self, A):
        combs = []
        ix = np.arange(self._metadata.n_particles)
        ix = np.delete(ix, A)
        for i in range(ix.shape[0]+1):
            combs.extend(combinations(ix,i))
        return np.array(combs)
    
    def proj(self, qubit:int, dir:object)->object:
        if(not isinstance(dir, TQobj)):
            raise TypeError('Must be TQobj')
        I = TQobj(torch.eye(self._metadata.hilbert_space_dims, self._metadata.hilbert_space_dims), n_particles=1, hilbert_space_dims=self._metadata.hilbert_space_dims)
        O =  [I for i in range(self._metadata.n_particles)]
        if(dir._metadata.obj_tp == 'operator'):
            O[qubit] = dir
        elif(dir._metadata.obj_tp == 'ket'):
            O[qubit] = dir @ dir.dag()
        else:
            O[qubit] = dir.dig() @ dir      
        O = direct_prod(*tuple(O)) 
        if self._metadata.obj_tp == 'operator':
            return O @ self @ O.dag()
        elif(self._metadata.obj_tp == 'ket'):
            return O @ self
        else:
            return self @ O.dag()
    
    def __getitem__(self, index):
        item = super(TQobj, self).__getitem__(index)
        try:
            item._metadata = self._metadata
        except:
            pass
        return item
    
    def sum(self, **kwargs):
        M = super(TQobj, self).sum(**kwargs)
        try:
            M.set_meta(self._metadata)
        except:
            M = M.to_tensor()
        return M
    
class TQobjEvo(Tensor):
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
    

#@nb.jit(nopython=False, forceobj=True)
def direct_prod(*args:tuple[TQobj])->TQobj:
    A = args[0]
    if(not isinstance(A, TQobj)):
        A = A[0]
        args = args[0]
        if(not isinstance(A, TQobj)):
            raise TypeError('Must be TQobj')
    
    m = A._metadata.n_particles
    h = A._metadata.hilbert_space_dims
    A = A
    for i, a in enumerate(args[1:]):
        if(isinstance(a, TQobj)):
            try:
                A = kron(A ,a)
                m+=a._metadata.n_particles
            except:
                ValueError('Must Have Particle Number')
        else:
            raise TypeError('Must be TQobj')
    meta = QobjMeta(n_particles=m, hilbert_space_dims=h, shp=A.shape)
    return TQobj(A, n_particles=m, hilbert_space_dims=h, meta = meta)

@torch.jit.script
def cummatprod_(O: Tensor) -> Tensor:
    for i in range(1, O.size(0)):
        O[i] = O[i] @ O[i-1]
    return O

@torch.jit.script
def ptrace_loc_ix(ix:Tensor, p:TQobj)->TQobj:
    if(len(p.shape) == 2):
        pA = torch.zeros((ix.shape[0], ix.shape[0]), dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] = p[ix[i], ix[j]].sum()
    else:
        pA = torch.zeros((p.shape[0], ix.shape[0], ix.shape[0]), dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA
