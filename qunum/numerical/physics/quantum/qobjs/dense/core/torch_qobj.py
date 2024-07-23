import torch
from numpy.typing import NDArray
from ...meta.meta import QobjMeta
from typing import Iterable, Self, Any
import copy
from torch import Tensor
from .core import ptrace_torch_ix as ptrace_ix, pT_arr, ventropy, cummatprod_, matprodcontract_
import numpy as np 
from itertools import combinations, product
from warnings import warn



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
                 **kwargs)->Self:
        #obj = super(TQobj,cls).__new__(cls, data,*args, dtype = torch.complex64,**kwargs)
        if('requires_grad' not in kwargs):
            try:
                kwargs['requires_grad'] = data.requires_grad
            except:
                kwargs['requires_grad'] = False

        if(isinstance(data, np.ndarray)):
            data = torch.from_numpy(data).to(dtype = dtype).requires_grad_(kwargs['requires_grad'])
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
                 is_hermitian:bool = False,
                 **kwargs)->Self:
        super(TQobj, self).__init__()
        self.set_meta(meta=meta, n_particles=n_particles,hilbert_space_dims=hilbert_space_dims, dims=dims, sparsify=sparsify, is_hermitian=is_hermitian)
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
                 is_hermitian:bool = False,
                 inplace:bool = True)->None:
        
        if(meta is None):
            meta = QobjMeta(
                n_particles=n_particles, 
                hilbert_space_dims=hilbert_space_dims,
                dims = dims,
                shp = self.shape,
                is_hermitian=is_hermitian
             )
        else:
            meta._reset_()
            meta.shp = self.shape
        if(inplace):
            self._metadata = meta
            return 
        else:
            O = self.to_tensor().clone()
            return TQobj(O, meta = meta)
    
    def abs_sqr(self)->Self:
        return self.conj()*self

    def conj(self)->Self:
        a = super(TQobj, self).conj()
        a.set_meta(self._metadata)
        return a

    def clone(self)->Self:
        a = super(TQobj, self).clone()
        a.set_meta(self._metadata)
        return a

    def __matmul__(self, O:Self|Tensor)->Self:
        assert isinstance(O, TQobj) or isinstance(O, Tensor), TypeError('Must Be TQobj or Tensor')
        M:Self = super(TQobj, self).__matmul__(O)
        M.set_meta(meta = QobjMeta(n_particles=self._metadata.n_particles, hilbert_space_dims=self._metadata.hilbert_space_dims, dims = self._metadata.dims, shp=M.shape))
        return M
    
    def __mul__(self, O:Self|Tensor|float|int)->Self:
        M = super(TQobj, self).__mul__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
        return M
    
    def __add__(self, O:Self|Tensor|float|int)->Self:
        M = super(TQobj, self).__add__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
        return M
    
    def __sub__(self, O:Self|Tensor|float|int)->Self:
        M = super(TQobj, self).__sub__(O)
        try:
            M.set_meta(meta = self._metadata)
        except:
            pass
        return M
    
    def __radd__(self, O:Self|Tensor)->Self:
        return self.__add__(O)
    
    def __rsub__(self, O:Self|Tensor)->Self:
        return self.__sub__(O)
    
    def __rmatmul__(self, O:Self|Tensor)->Self:
        assert isinstance(O, TQobj) or isinstance(O, Tensor), TypeError('Must Be TQobj or Tensor')
        U:Self = O.__matmul__(self)
        U.set_meta(meta = QobjMeta(n_particles=self._metadata.n_particles, hilbert_space_dims=self._metadata.hilbert_space_dims, dims = self._metadata.dims, shp=U.shape))
        return U
    
    def __rmul__(self, O:Self|Tensor)->Self:
        return self.__mul__(O)
    
    def __truediv__(self, O:Self|Tensor)->Self:
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
    
    def __xor__(self, O:Self)->Self:
        return direct_prod(self,O)
    
    def __rxor__(self, O:Self) -> object:
        return direct_prod(O,self)

    def to_sparse_qobj(self)->None:
        raise NotImplementedError('Not Yet Implemented ')
    
    def zero_grad(self)->None:
        try:
            self.grad.data.zero_()
        except:
            pass
        return 
    
    def get_grad(self)->Tensor|Self:
        self.retain_grad()
        return self.grad
    
    def dag(self)->Self:
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
    
    def expm(self, recompute_eig:bool = False)->Self:
        V, U = self.diagonalize(inplace=False, ret_unitary=True, save_eigen=False, recompute=recompute_eig)
        return U @ V.exp() @ U.dag()

    def logm(self, recompute_eig:bool = False)->Self:
        l, P = self.diagonalize(inplace=False, ret_unitary=True, save_eigen=False, recompute=recompute_eig)
        return (P @ l.log() @ P.dag())
    
    def eig(self,*args, eigenvectors:bool = True, save:bool = False, recompute:bool = True, check_hermitian:bool = False, **kwargs)->tuple[Tensor, Self]|Tensor:
        def getEig()->tuple[TQobj, Tensor]|Tensor:
            if((self._metadata.eigenVals is None or (self._metadata.eigenBasis is None and eigenvectors)) or recompute):
                match (int(eigenvectors), int(self._metadata.is_hermitian)):
                    case (1,1):
                        v, U = torch.linalg.eigh(self, *args, **kwargs)
                    case (1,0):
                        v, U = torch.linalg.eig(self, *args, **kwargs)
                       
                        
                    case (0,1):
                        v:Tensor = torch.linalg.eigvalsh(self, *args, **kwargs)
                        U = None
                    case _:
                        v:Tensor = torch.linalg.eigvals(v, *args, **kwargs)
                        U = None
                if(save):
                    self._metadata.eigenVals = v.to_sparse()
                    if(U is not None):
                        self._metadata.eigenBasis = U.to_sparse()
                return v, TQobj(U, meta=self._metadata)
            elif(eigenvectors):
                return self._metadata.eigenVals.to_dense(), TQobj(self._metadata.eigenBasis.to_dense(), meta = self._metadata)
            else:
                return self._metadata.eigenVals.to_dense()
        if(check_hermitian):
            self.check_herm()
        return getEig()

    def diagonalize(self, *args, inplace:bool = False, ret_unitary:bool=False, save_eigen:bool = False, recompute:bool = False, **kwargs) -> tuple[Self,Self]|Self|None:
        assert (self._metadata.obj_tp == 'operator'), TypeError('Must be operator type')
        v, U = self.eig(*args, eigenvectors=ret_unitary, save=save_eigen, recompute=recompute, **kwargs) 
        match (inplace, ret_unitary):
            case (True, True):
                self.data = torch.diag_embed(v)
                return U 
            case (True, False):
                self.data = TQobj(torch.diag_embed(v), meta= self._metadata)
                return 
            case (False, True):
                return TQobj(torch.diag_embed(v), meta= self._metadata), U
            case (False, False):
                return TQobj(torch.diag_embed(v), meta= self._metadata)
        return
    
    def check_herm(self)->bool:
        assert self._metadata.obj_tp == 'operator', TypeError('Must be an operator')
        self._metadata.is_hermitian = (self == self.dag()).all()
        return self._metadata.is_hermitian

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
    
    def pT(self, ix_:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable[int])->Self:
        assert len(self.shape)<3, ValueError('Not implemented for shapes larger than 2')
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        ix_ =  torch.tensor(self._metadata.query_particle_ixs(self._metadata.check_particle_ixs(ix_)))[:,0]
        return TQobj(pT_arr(torch.tensor(self), ix_), meta = self._metadata)
    
    def entropy(self, tp_calc:str ='von', n_reyni:int = 2, von_epsi:float = 1e-8, **kwargs)->Self:
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        match tp_calc:
            case 'von':
                return ventropy(self,epsi=von_epsi)
            case 'reyni':
                return reyni_entropy(self,n=n_reyni)
            case _:
                warn('Error tp_cal must be in {"von", "reyni"} Only Von Nueman and Reyni entropy implemented assuming Von Neumann Entory')
                return ventropy(self, epsi = von_epsi)
                 
    def polarization_vector(self, particle:int)->torch.Tensor:
        from ......mathematics.algebra.representations import su
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
    
    def to(self, *args, **kwargs)->Self:
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

    def mutual_info(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, B:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None = None)->Self:
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
        from ......mathematics.algebra.representations import su
        if(self._metadata.hilbert_space_dims != 2):
            raise ValueError('To pauli decompose must be 2d hilbert spaces ')
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
        from ......mathematics.algebra import commutator as comm
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
    
    def pidMatrix(self, A:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable, Projs:Self)->Tensor:
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
    
    def cummatprod(self, left_or_right:str = 'left'):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must Be operator type')
        if(len(self.shape) != 3):
            raise TypeError('Must Be operator type with multiple entries rank(3)')
        return TQobj(cummatprod_(self.to_tensor(), left_or_right=left_or_right), meta = self._metadata)
    
    def matprodcontract(self, left_or_right:str = 'left'):
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must Be operator type')
        if(len(self.shape) != 3):
            raise TypeError('Must Be operator type with multiple entries rank(3)')
        return TQobj(matprodcontract_(self.to_tensor(), left_or_right=left_or_right), meta = self._metadata)
        
    def get_systems(self, A:Iterable[int]|int)->Iterable[int]:
        combs = []
        ix = np.arange(self._metadata.n_particles)
        ix = np.delete(ix, A)
        for i in range(ix.shape[0]+1):
            combs.extend(combinations(ix,i))
        return np.array(combs)
    
    def proj(self, qubit:int, dir:Self)->Self:
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
    
    def __getitem__(self, index)->Self:
        item:Self = super(TQobj, self).__getitem__(index)
        try:
            item._metadata = self._metadata
        except:
            pass
        return item
    
    def sum(self, **kwargs)->Self:
        M:Self = super(TQobj, self).sum(**kwargs)
        try:
            M.set_meta(self._metadata)
            M._metadata.shp = M.shape
        except:
            pass
        return M
    
    def cumsum(self, **kwargs)->Self:
        M:Self = super(TQobj, self).cumsum(**kwargs)
        try:
            M.set_meta(self._metadata)
            M._metadata.shp = M.shape
        except:
            pass
        return M
    
    def mat_pow(self, n:int|float)->Self:
        M:Self = torch.linalg.matrix_power(self, n)
        M.set_meta(self._metadata)
        return M

    def to(self,*args:tuple[Any], obj_tp:None|str=None, **kwargs:dict[str:Any])->Self:
        d:Self = super(TQobj, self).to(*args, **kwargs)
        d.set_meta(meta = self._metadata)
        return d

    def toSuN(self)->Self:
        pass

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