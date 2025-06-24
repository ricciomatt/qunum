import torch
from numpy.typing import NDArray
from ...meta.meta import QobjMeta
from typing import Iterable, Self, Any, Callable

import copy
from torch import Tensor, view_as_real as toR
from .core import ptrace_torch_ix as ptrace_ix, pT_arr, ventropy, cummatprod_, matprodcontract_
import numpy as np 
from itertools import combinations, product
from warnings import warn
from polars import from_numpy as pl_from_numpy, concat_str, col, when, lit, Int32, String, DataFrame as PlDataFrame, LazyFrame as PlLazyFrame, Series as PlSeries, format as plform
from pandas import DataFrame as PdDataFrame
from pyarrow import Table as PyArrowTable


'''Need to Fix the Gradient Function'''
class TQobj(Tensor):
    def __new__(cls, 
                data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int = 2,
                 dims:None|dict[str:int] = None,
                 dtype = torch.complex128,
                 **kwargs)->Self:
        #obj = super(TQobj,cls).__new__(cls, data,*args, dtype = torch.complex64,**kwargs)
        if('requires_grad' not in kwargs):
            if(isinstance(data, Tensor)):
                kwargs['requires_grad'] = data.requires_grad_
            else:
                kwargs['requires_grad'] = False
        elif(isinstance(data,list)):
            data = np.array(data)
        if(isinstance(data, np.ndarray)):
            data = torch.from_numpy(data).to(dtype = dtype).requires_grad_(kwargs['requires_grad'])
        elif(data.dtype != dtype):
            data = data.to(dtype)
        assert isinstance(data, Tensor), TypeError('Must be List, Numpy Array, or torch Tensor')
        obj = super(TQobj, cls).__new__(cls, data)
        return obj
    
    def __init__(self, 
                 data,
                 *args,
                 meta:QobjMeta|None = None, 
                 n_particles:int|None = 1, 
                 hilbert_space_dims:int= 2,
                 dims:None|dict[str:int] = None,
                 is_hermitian:bool = False,
                 **kwargs)->Self:
        super(TQobj, self).__init__()
        self.set_meta(meta=meta, n_particles=n_particles,hilbert_space_dims=hilbert_space_dims, dims=dims,  is_hermitian=is_hermitian)
        if(self.requires_grad):
            self.retain_grad()
        #if(sparsify):
         #   self.to_sparse_qobj()
        return
    
    #--Core--
    
    def set_meta(
            self,
            meta:QobjMeta|None = None, 
            n_particles:int = 1, 
            hilbert_space_dims:int =2,
            dims:None|dict[int:set] = None ,
            is_hermitian:bool = False,
            inplace:bool = True
        )->None:
        
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
            return TQobj(O, meta = copy.copy(meta))
    
    def dag(self)->Self:
        meta = copy.copy(self._metadata)
        if(self._metadata.obj_tp == 'ket'):
            meta.obj_tp = 'bra'
        elif(self._metadata.obj_tp == 'bra'):
            meta.obj_tp = 'ket'
        M:TQobj = self.conj().swapaxes(-1,-2)
        M.set_meta(meta= meta)
        return M

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
    
    def eig(self,*args, eigenvectors:bool = True, save:bool = False, recompute:bool = True, check_hermitian:bool = False, hermitian:bool = False, ret_tensor:bool= False, **kwargs)->tuple[Tensor, Self]|Tensor:
        def getEig()->tuple[TQobj, Tensor]|tuple[Tensor, Tensor]|Tensor:
            if((self._metadata.eigenVals is None or (self._metadata.eigenBasis is None and eigenvectors)) or recompute):
                match (int(eigenvectors), int(self._metadata.is_hermitian or hermitian)):
                    case (1,1):
                        v, U = torch.linalg.eigh(self.to_tensor(), *args, **kwargs)
                    case (1,0):
                        v, U = torch.linalg.eig(self.to_tensor(), *args, **kwargs)
                    case (0,1):
                        v:Tensor = torch.linalg.eigvalsh(self.to_tensor(), *args, **kwargs)
                        U = None
                    case _:
                        v:Tensor = torch.linalg.eigvals(self.to_tensor(), *args, **kwargs)
                        U = None
                if(save):
                    if(isinstance(U,Tensor)):
                        self._metadata.set_eig(v,U.conj().swapdims(-1,-2))
                    else:
                        self._metadata.set_eig(v,None)
                if(U is None):
                    return v
                elif(ret_tensor):
                    return v, U.conj().swapdims(-1,-2)
                else:
                    return v, TQobj(U.conj().swapdims(-1,-2), meta=self._metadata)
            elif(eigenvectors):
                v, U = self._metadata.get_eig(eigenvectors)
                if(ret_tensor):
                    return v, U
                else:
                    return v, TQobj(U, meta = self._metadata)
            else:
                return self._metadata.get_eig_vals()
        if(check_hermitian):
            self.check_herm()
        return getEig()

    def to_tensor(self, detach=True, inplace=False)->Tensor:
        if(inplace):
            self.__class__ = torch.Tensor
            del self._metadata
            return 
        else:
            A = self.clone(to_tensor=True)
            if(detach):
                return A.detach()
            else:
                return A
               
    # -- Override Methods --

    def inverse(self)->Self:
        return self.inv(inplace=False)

    def swapdims(self, dim0=-1, dim1=-2)->Self:
        T:TQobj = super(TQobj, self).swapdims(dim0=dim0, dim1=dim1)
        T.set_meta(meta = self._metadata)
        return T
    
    def swapaxis(self, axis0=-1, axis1=-2)->Self:
        return super(TQobj, self).swapaxis(axis0=axis0, axis1= axis1).set_meta(meta=self.set_meta, inplace=False)

    def conj(self)->Self:
        a = super(TQobj, self).conj()
        a.set_meta(self._metadata)
        return a

    def clone(self, to_tensor:bool = False)->Self:
        a:TQobj = super(TQobj, self).clone()
        if(to_tensor):
            a.__class__= torch.Tensor
            return a
        else:
            a.set_meta(self._metadata)
            return a
        
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
    
    def to(self,*args:tuple[Any], obj_tp:None|str=None, **kwargs:dict[str:Any])->Self:
        d:Self = super(TQobj, self).to(*args, **kwargs)
        d.set_meta(meta = self._metadata)
        return d
    
    def zero_grad(self)->None:
        try:
            self.grad.data.zero_()
        except:
            pass
        return 
    
    def get_grad(self)->Tensor|Self:
        self.retain_grad()
        return self.grad
   
    def cpu(self,)->None:
        return self.to('cpu')
    
    # -- Operator Overloads --

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
        M:TQobj = super(TQobj, self).__div__(O)
        M.set_meta(meta= self._metadata)
        return M

    def __rtuediv__(self, O:Self|Tensor):
        assert isinstance(O,TQobj) or isinstance(O,Tensor), TypeError('Must be Tensor or TQobj')
        M:TQobj = super(TQobj, self).__rdiv__(O)
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
        # Direct Product Space
        return direct_prod(self,O)
    
    def __rxor__(self, O:Self) -> Self:
        # Direct Product Space
        return direct_prod(O,self)

    def __array__(self)->np.ndarray:
        return self.detach().numpy()
    
    def __getitem__(self, *index:tuple[int|slice|range|list|Tensor|NDArray, ...])->Self:
        item:Self = super(TQobj, self).__getitem__(*index)
        try:
            item._metadata = self._metadata
        except:
            pass
        return item
    
    # -- Functions of Matricies --
    
    def expm(self, recompute:bool = False, save:bool = False, check_hermitian:bool = False, method:str|None=None)->Self:
        if(method is None):
            if(self._metadata.eigenVals is not None and self._metadata.eigenBasis is not None):
                method = 'eig'
                df = 'eig'
            else:
                method = 'std'
                df= 'std'
        def mtch(method:str)->TQobj:
            match method:
                case 'std':
                    M:TQobj = torch.linalg.matrix_exp(self)
                    return M.set_meta(meta=self._metadata, inplace = False)
                case 'eig':
                    V, U = self.eig(recompute=recompute, eigenvectors = True, save = save , check_hermitian=check_hermitian)
                    return U.dag() @ TQobj((V.exp().diag_embed()),meta=self._metadata) @ U
                case _:
                    warn(KeyError('method Must be in "std" or "eig" defaulting to {df}'.format(df= df)))
                    return mtch(df)
        return mtch(method)
                    
    def mat_pow(self, n:int|float|Tensor, recompute:bool = False, save:bool = False, check_hermitian:bool = False, method:str|None = None)->Self:
        if(method is None):
            if(self._metadata.eigenVals is not None and self._metadata.eigenBasis is not None):
                method = 'eig'
                df = 'eig'
            else:
                method = 'std'
                df= 'std'
        def mtch(method:str)->TQobj:
            match method:
                case 'std':
                    M:TQobj = torch.linalg.matrix_power(self, n)
                    return M.set_meta(meta=self._metadata, inplace = False)
                case 'eig':
                    V, U = self.eig(recompute=recompute, eigenvectors = True, save = save , check_hermitian=check_hermitian)
                    return U.dag() @ TQobj((V.pow(n).diag_embed()),meta=self._metadata) @ U
                case _:
                    warn(KeyError('method Must be in "std" or "eig" defaulting to {df}'.format(df= df)))
                    return mtch(df)
        return mtch(method)

    def inv(self, inplace:bool = False, recompute:bool = False, save:bool = False, check_hermitian:bool = False, method:str|None=None, **kwargs)->Self:
        if(method is None):
            if(self._metadata.eigenVals is not None and self._metadata.eigenBasis is not None):
                method = 'eig'
                df = 'eig'
            else:
                method = 'std'
                df= 'std'
        def mtch(method)->TQobj:
            match method:
                case 'std':
                    return super(TQobj, self).inverse()
                case 'tensorinv':
                    return torch.linalg.tensorinv(self,**kwargs) 
                case 'eig':
                    V,U = self.eig(eigenvectors=True, save=save, recompute=recompute, check_hermitian=check_hermitian)
                    return U.dag() @ TQobj((V**-1).diag_embed(), meta= self._metadata) @ U
                case _: 
                    warn(KeyError('method Must be in "std" or "eig" defaulting to {df}'.format(df= df)))
                    return mtch(df)
        if(inplace):
            self.data = mtch(method)
        else: 
            return mtch(method).set_meta(inplace = False, meta= self._metadata) 

    def logm(self, recompute:bool = False, save:bool = False, check_hermitian:bool = False)->Self:
        V, U = self.eig(recompute=recompute, eigenvectors = True, save = save , check_hermitian=check_hermitian)
        return U.dag() @ TQobj(V.log().diag_embed(), meta=self._metadata) @ U

    def logbasem(self, base:Tensor = torch.tensor(torch.e), recompute:bool = False, save:bool = False, check_hermitian:bool = False)->Self:
        V, U = self.eig(recompute=recompute, eigenvectors = True, save = save , check_hermitian=check_hermitian)
        return U.dag() @ TQobj((V.log()/base.log()).diag_embed(), meta=self._metadata) @ U
    
    def applyFToMatrix(self, fn:Callable[[Self|Tensor,tuple, dict], Self], *args, f_of_lambda:bool = True, vmaped:bool= False, **kwargs)->Self:
        assert self._metadata.obj_tp=='operator','Must be operator type to compute f(O)'
        warn('Best practicize use \n@qunum.numerical.physics.qunatum.qobjs.dense.fofMatrix(*args, **kwargs)\ndef some_function(A,*args,**kwargs):\n...')
        from ..linalg.core import fofMatrix
        return fofMatrix(vmaped=vmaped, f_of_lambda=f_of_lambda, **kwargs)(fn)(self, *args, **kwargs)

    # -- Quantum Operations --
    
    def get_systems(self, A:Iterable[int]|int)->Iterable[int]:
        combs = []
        ix = np.arange(self._metadata.n_particles)
        ix = np.delete(ix, A)
        for i in range(ix.shape[0]+1):
            combs.extend(combinations(ix,i))
        return np.array(combs)
        
    def Tr(self, tr_out:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, keep:list[int]|tuple[int]|NDArray|Tensor|slice|int|Iterable|None=None, reorder:bool = False, **kwargs)->torch.Tensor|Self:
        from ......mathematics.combintorix import EnumerateArgCombos
        if(self._metadata.obj_tp != 'operator'):
            raise TypeError('Must be an operator')
        if(tr_out is None and keep is None):
            return self.diagonal(dim1=-2, dim2=-1).sum(dim=-1).to_tensor()
        else:
            if(tr_out is not None):
                tr_out = self._metadata.check_particle_ixs(tr_out)
                ix:NDArray = np.arange(self._metadata.n_particles)
                ix:NDArray = np.delete(ix, tr_out)
                
            else:
                ix:NDArray = self._metadata.check_particle_ixs(keep)
            if(ix.shape[0] == self._metadata.n_particles):
                return self
            
            ix_ =  torch.from_numpy(self._metadata.query_particle_ixs(ix)).T
            lix = ix_.shape[0]
            ix_ = EnumerateArgCombos(ix_, ix_).__tensor__()
            meta = copy.copy(self._metadata)
            meta.update_dims(ix, reorder=reorder)
            if('debug' in kwargs):
                print(ix_, lix,ix)
                print(self[..., ix_[:,0], ix_[:,1]])
            
            return self[...,ix_[:,0], ix_[:,1]].sum(dim=-1).reshape((*self.shape[:-2],lix,lix)).set_meta(meta,inplace=False)
            
        
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
        from ......mathematics.algebra import sun
        assert self._metadata.obj_tp == 'operator', TypeError('Must be an operator')
        assert self._metadata.dims[particle] == 2, ValueError('PVec Support Limited to only the SU(2) case for now')
        p = self.Tr(keep=particle).to_tensor().to(self.dtype)
        s = sun.get_pauli(to_tensor= True).to(self.dtype)
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
        from ......mathematics.algebra import sun
        if(self._metadata.hilbert_space_dims != 2):
            raise NotImplementedError('Only implemented for 2d hilbert spaces')
        sig = TQobj(sun.get_pauli(to_tensor=True), n_particles=1)
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
    
    def abs_sqr(self)->Self:
        return self.conj()*self
        
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

    def to_braket(self,  to_ = 'PdDataFrame') -> PdDataFrame|PlDataFrame|PlLazyFrame|np.ndarray[str]|PyArrowTable|PlSeries|dict[str:PlSeries]|list[dict[str:str]]|str:
        assert len(self.shape) == 2, AssertionError('Can only output braket notation for signle Qobj Elements with len(Obj.shape) == 2, call Obj[0].to_braket(to_="str")')
        ExtractCoef = when(
            ((col(f'Re')!= 0) | (col(f'Im') != 0))
        ).then(
            when(
                ((col(f"Re")!=0) & (col(f'Im') == 0))
            ).then(
                plform('{}', col(f"Re").round_sig_figs(3))
            ).when(
                ((col(f"Re")!=0) & (col(f'Im') > 0))
            ).then(
                '('+ plform('{}', col(f"Re").round_sig_figs(3))+'+'+ plform('{}', col(f"Im").abs().round_sig_figs(3))+f'i)'
            ).when(
                ((col(f"Re")!=0) & (col(f'Im') < 0))
            ).then(
                '('+ plform('{}', col(f"Re").round_sig_figs(3))+'-'+plform("{}",col(f'Im').abs().round_sig_figs(3))+f'i)'
            ).when(
                ((col(f"Re")==0) & (col(f'Im') != 0))
            ).then(
                plform('{}', col(f"Im").round_sig_figs(3))+f'i'
            )
        ).alias('Coef')
        match self._metadata.obj_tp:
            case 'ket':
                def to_braket()->PlLazyFrame:
                    data = toR(self[:,0].to_tensor())
                    return pl_from_numpy(data.detach().numpy(), schema=['Re', 'Im']).lazy().with_row_index().join(self._metadata.ixs, on='index').with_columns(ExtractCoef).with_columns((col('Coef')+'\\left|'+ concat_str([f'column_{d}' for d in self._metadata.dims])+'\\right>').alias('BraKet')).select('BraKet')        
            case 'bra':
                def to_braket()->PlLazyFrame:
                    data = toR(self[0].to_tensor())
                    return pl_from_numpy(data.detach().numpy(), schema=['Re', 'Im']).lazy().with_row_index().join(self._metadata.ixs, on='index').with_columns(ExtractCoef).with_columns(('\\left<'+ concat_str([f'column_{d}' for d in self._metadata.dims])+'\\right|'+col('Coef')).alias('BraKet')).select('BraKet') 
            case "operator":
                raise NotImplementedError('Not yet Implemented for operator')
            case _:
                raise TypeError('Must be ket or bra to show braket') 
        Data = to_braket()
        match to_.lower():
            case to_ if to_ in ['str', 'latex']:
                return f"$${' + '.join(Data.collect().to_numpy()[:,0])}$$"
            case 'array':
                return Data.collect().to_numpy()
            case _:
                Data = Data.with_columns('$$'+col('BraKet')+'$$')
        match to_.lower():
            case 'pyarrowtable':
                return Data.collect().to_arrow()
            case 'dict':
                return Data.collect().to_dict()
            case 'dicts':
                return Data.collect().to_dicts()
            case 'pddataframe':
                return Data.collect().to_pandas()
            case 'pldataframe':
                return Data.collect()
            case 'pllazyframe':
                return Data
            case 'plseries':
                return Data.collect().to_series()    
            case _:
                warn('Returning Lazy Frame to_ should be in {PdDataFrame,PlDataFrame,PlLazyFrame, array, str, PyArrowTable, dict,dicts,PlSeries, LaTeX}')
                return Data
        
@torch.jit.script
def ptrace_bwd_ix(ix:torch.Tensor, p:TQobj, pA:TQobj)->TQobj:
    if(len(p.shape) == 2):
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] += p[ix[i], ix[j]].sum()
        return pA
    else:
        pA = pA.swapdims(0,-2).swapdims(1,-1)
        p = p.swapdims(0,-2).swapdims(1,-1)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] += p[ix[i], ix[j]].sum(dim = [0])
        return pA.swapdims(1,-1).swapdims(0,-2)


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

def int_to_bin(n:int, m:int):
    return format(n, '0{m}b'.format(m  = m))


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