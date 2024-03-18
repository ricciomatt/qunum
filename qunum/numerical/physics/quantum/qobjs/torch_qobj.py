import torch
import polars as pl
from numpy.typing import NDArray
from .meta import QobjMeta
from typing import Any, Iterable
import copy
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
from .density_operations import ptrace_torch_ix as ptrace_ix, vgc, nb_get_cols, pT_arr, ventropy, cummatprod_
from warnings import warn
from IPython.display import display as disp, Markdown as md, Math as mt
from torch import Tensor
import numpy as np 
from torch import kron
import numba as nb
from itertools import combinations, product
from scipy.linalg import logm



'''Need to Fix the Gradient Function'''
class TQobj(Tensor):
    def __new__(cls, 
                data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int = 2,
                 sparsify:bool = True,
                 dims:None|dict[str:int] = None,
                 dtype = torch.complex128,
                 **kwargs):
        #obj = super(TQobj,cls).__new__(cls, data,*args, dtype = torch.complex64,**kwargs)
        if('requires_grad' not in kwargs):
            try:
                kwargs['requires_grad'] = data.requires_grad
            except:
                kwargs['requires_grad'] = False

        if(isinstance(data, np.ndarray)):
            data = torch.from_numpy(data, *args, dtype=dtype, **kwargs)
        
        elif(isinstance(data, list)):
            try:
                data = torch.tensor(data, dtype = dtype)
            except Exception as E:
                raise TypeError(E)
        elif(data.dtype != dtype):
            data = data.to(dtype)
        assert isinstance(data, Tensor), TypeError('Must be Numpy Array or Tensor Type')
        if(sparsify):
            pass
            #data.to_sparse()
        obj = super(TQobj, cls).__new__(cls, data)
        return obj
    
    def __init__(self, 
                 data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int= 2,
                 dims:None|dict[str:int] = None,
                 sparsify:bool = False,
                 **kwargs)->object:
        super(TQobj, self).__init__()
        self.set_meta(meta=meta, n_particles=n_particles,hilbert_space_dims=hilbert_space_dims, dims=dims, sparsify=sparsify)
        if(self.requires_grad):
            self.retain_grad()
        #if(sparsify):
         #   self.to_sparse_qobj()
        return
    
    def set_meta(self,
                 meta:QobjMeta|None = None, 
                 n_particles:int = 1, 
                 hilbert_space_dims:int =2,
                 dims:None|dict[int:set] = None ,
                 sparsify:bool = True,):
        if(meta is None):
            self._metadata = QobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                dims = dims,
                shp = self.shape
             )
        else:
            meta._reset_()
            self._metadata = meta
        return 
    
    def abs_sqr(self)->object:
        return self.conj()*self

    def conj(self)->object:
        a = super(TQobj, self).conj()
        a.set_meta(self._metadata)
        return a
    
    def diagonalize(self, full_inversion:bool = False)->object:
        if(self._metadata.eigenBasis is None):
            self._metadata.eigenVals, self._metadata.eigenBasis = torch.linalg.eig(self)
        if(full_inversion):
            return self._metadata.eigenBasis @ self @ torch.linalg.inv(self._metadata.eigenBasis)
        else:
            return self._metadata.eigenBasis @ self  @ self._metadata.eigenBasis.conj().T

    def clone(self)->object:
        a = super(TQobj, self).clone()
        a.set_meta(self._metadata)
        return a

    def __matmul__(self, O:object|Tensor)->object:
        if not (isinstance(O, TQobj) or isinstance(O,Tensor)):
            raise TypeError('Must Be TQobj or Tensor')
        M = super(TQobj, self).__matmul__(O)
        M.set_meta(meta = QobjMeta(n_particles=self._metadata.n_particles, hilbert_space_dims=self._metadata.hilbert_space_dims, dims = self._metadata.dims, shp=M.shape))
        return M
    
    def __mul__(self, O:object|Tensor|float|int)->object:
        M = super(TQobj, self).__mul__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
        return M
    
    def __add__(self, O:object|Tensor|float|int)->object:
        M = super(TQobj, self).__add__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
        return M
    
    def __sub__(self, O:object|Tensor|float|int)->object:
        M = super(TQobj, self).__sub__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
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
        M = super(TQobj, self).__div__(O)
        M.set_meta(meta= self._metadata)
        return M

    
    def __repr__(self)->str:
        try:
            str_ = self._metadata.__str__()
        except:
            str_ = 'No dataset available'
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
        M = torch.linalg.matrix_exp(self)
        M.set_meta(meta = self._metadata)
        return M

    def logm(self, invert_:bool = False)->object:
        l, P = self.eig()
        D = torch.zeros_like(self)
        if(invert_):
            m = P._metadata
            PI = TQobj(torch.linalg.inv(P.to_tensor()), meta=m)
        PI = P.dag()
        D[..., torch.arange(self.shape[-1]), torch.arange(self.shape[-1])] = torch.log(l).squeeze()
        return (P @ TQobj(D,meta = self._metadata) @ PI)
    
    def eig(self, eigenvectors:bool = True, save:bool = False, recompute:bool = True)->tuple[Tensor, object]:
        if(eigenvectors):
            if(self._metadata.eigenBasis is None or recompute):
                l, P = torch.linalg.eig(self.to_tensor())
                if(save):
                    self._metadata.eigenVals,self._metadata.eigenBasis  = l.to_sparse(), P.to_sparse()
                P = TQobj(P,meta = self._metadata)
            else:
                l,P = self._metadata.eigenVals.to_dense(), self._metadata.eigenBasis.to_dense()
                P = TQobj(P, meta = self._metadata)
            return l, P
        else:
            if(self._metadata.eigenVals is None or recompute):
                l = torch.linalg.eigvals(self.to_tensor())
                if(save):
                    self._metadata.eigenVals = l
                P = TQobj(P,meta = self._metadata)
            else:
                l,P = self._metadata.eigenVals.to_dense(), self._metadata.eigenBasis.to_dense()
                P = TQobj(P, meta = self._metadata)
            return l, P
        

    def to_tensor(self, detach=True)->Tensor:
        if(detach or self.requires_grad == False):
            try:
                return torch.from_numpy(self.data.detach().numpy()).to(self.dtype)
            except:
                return torch.from_numpy(self.data.detach().resolve_conj().numpy()).to(self.dtype)
    
    def Tr(self, tr_out:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, keep:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, reorder:bool = False):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        if(tr_out is None and keep is None):
            return self.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        else:
            if(tr_out is not None):
                tr_out = self._metadata.check_particle_ixs(tr_out)
                ix = np.arange(self._metadata.n_particles)
                ix = np.delete(ix, tr_out)
            else:
                ix = self._metadata.check_particle_ixs(keep)
            if(ix.shape[0] == self._metadata.n_particles):
                return self
            ix_ =  torch.tensor(self._metadata.query_particle_ixs(ix))[:,0]
            meta = copy.copy(self._metadata)
            meta.update_dims(ix, reorder=reorder)
        if(self.requires_grad):
            if(len(self.shape) > 2):
                pA = TQobj(torch.zeros((self.shape[0], ix_.shape[0], ix_.shape[0]), dtype=self.dtype), meta = meta)
            else:
                pA = TQobj(torch.zeros((ix_.shape[0], ix_.shape[0]), dtype=self.dtype), meta = meta)
            return ptrace_bwd_ix(ix_, self, pA)
        else:
            A = TQobj(ptrace_ix(ix_, self.clone().detach()), meta = meta) 
            A._metadata.shp = A.shape
            return  A
    
    def pT(self, ix_T:tuple[int]|list[int])->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        ix_ =  torch.tensor(self._metadata.query_particle_ixs(ix_))[:,0]
        return TQobj(pT_arr(torch.tensor(self), ix_), meta = self._metadata)
    
    def entropy(self, tp_calc:str ='von', n_reyni:int = 2, von_epsi:float = 1e-8, **kwargs)->object:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        entropy_map = {'von':ventropy, 'reyni':reyni_entropy}
        if(tp_calc not in entropy_map):
            return ventropy(self, epsi=von_epsi)
        else:
            if(tp_calc == 'von'):
                return entropy_map[tp_calc](self,epsi=von_epsi)
            else:
                return entropy_map[tp_calc](self,n=n_reyni)
            
    
    def polarization_vector(self, particle:int)->torch.Tensor:
        from ....mathematics.algebra.representations import su
        assert self._metadata.obj_tp == 'operator', TypeError('Must be an operator')
        assert self._metadata.dims[particle] == 2, ValueError('PVec Support Limited to only the SU(2) case for now')
        p = self.Tr(keep=particle).to_tensor().to(self.dtype)
        s = su.get_pauli(to_tensor= True).to(self.dtype)
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
    
    def to(self, *args, **kwargs)->object:
        self.device
        M = super(TQobj, self).to(*args, **kwargs)
        try:
            M.set_meta(self._metadata)
        except:
            try:
                M = M.to_tensor()
            except:
                pass
            pass
        return M
    
    def cpu(self,)->None:
        return self.to('cpu')

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
        from ....mathematics.algebra.representations import su
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
        from ....mathematics.algebra import commutator as comm
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
        
    def get_systems(self, A:Iterable[int]|int)->Iterable[int]:
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
    
    def __getitem__(self, index)->object:
        item = super(TQobj, self).__getitem__(index)
        try:
            item._metadata = self._metadata
        except:
            pass
        return item
    
    def sum(self, **kwargs)->object:
        M = super(TQobj, self).sum(**kwargs)
        try:
            M.set_meta(self._metadata)
            M._metadata.shp = M.shape
        except:
            pass
        return M
    
    def cumsum(self, **kwargs)->object:
        M = super(TQobj, self).cumsum(**kwargs)
        try:
            M.set_meta(self._metadata)
            M._metadata.shp = M.shape
        except:
            pass
        return M
    def mat_pow(self, n:int)->object:
        M = torch.linalg.matrix_power(self, n)
        M.set_meta(self._metadata)
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
    

@torch.jit.script
def ptrace_bwd_ix(ix:torch.Tensor, p:TQobj, pA:TQobj)->TQobj:
    if(len(p.shape) == 2):
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] += p[ix[i], ix[j]].sum()
    else:
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA

'''
@torch.jit.script
def ptrace_bwd_ix(ix:Tensor, p:TQobj)->TQobj:
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
def direct_prod(*args:tuple[TQobj])->TQobj:
    from string import ascii_uppercase
    A = args[0]
    assert isinstance(A, TQobj), f'Arguments must be Qunatum Objects(TQobj), item 0'
    dims = {ix:A._metadata.dims[d] for ix, d in enumerate(A._metadata.dims)}
    cixs = ''.join(map(lambda ix: ascii_uppercase[ix], range(len(A.shape[:-2]))))
    for i, a in enumerate(args[1:]):
        assert isinstance(a, TQobj), f'Arguments must be Qunatum Objects(TQobj), item {i}'
        assert isinstance(a._metadata, QobjMeta), f"Arguments must have valid Object Data, item {i}"     
        A = torch.flatten(torch.flatten(torch.einsum(f'{cixs}ij, {cixs}km->{cixs}ikjm', A, a), start_dim=-2), start_dim=-3, end_dim=-2)
        tdims = {len(dims)+ix:a._metadata.dims[d] for ix, d in enumerate(a._metadata.dims)}
        dims.update(tdims)
    meta = QobjMeta(dims=dims, shp=A.shape)
    return TQobj(A, meta = meta)


#@torch.jit.script
def reyni_entropy(p:TQobj, n:int=2)->TQobj:
    pn = p.mat_pow(n).Tr()
    return (1/(1-n))*torch.log(pn)

