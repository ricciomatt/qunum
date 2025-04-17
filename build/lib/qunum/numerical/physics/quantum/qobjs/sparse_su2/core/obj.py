import torch 
import numpy as np
from .core import addFun, subFun, doMul, ptrace, fullTrace, doMulStateMat, doMulStateState
from typing import Self, Iterable
from torch import view_as_real as toR, view_as_complex as toC
from polars import from_numpy as pl_from_numpy, concat_str, col, when, lit, Int32, String, DataFrame as PlDataFrame, LazyFrame as PlLazyFrame, Series as PlSeries, format as plform
from pandas import DataFrame as PdDataFrame
from pyarrow import Table as PyArrowTable
from warnings import warn
from typing import Callable

class State:
    def __init__(self, objTp:str = 'ket') -> Self:
        self.objTp:str= objTp
        return
    def set_obj_tp(self, objTp:str='ket')->Self:
        self.objTp = objTp
        return
    


class PauliMatrix:
    def __init__(
        self, basis:torch.Tensor, coefs:torch.Tensor, 
        time_dependence:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]|None = None, 
        spatial_projection:Callable[[torch.Tensor], torch.Tensor]|None = None, 
        is_zero:bool=False, 
        dtype:torch.dtype = torch.complex128, 
        device:torch.device = 'cpu')->Self:
        assert (basis is None and coefs is None) or isinstance(coefs,torch.Tensor) and isinstance(basis, torch.Tensor), TypeError('Must be Tensor type ')
        self.dtype = dtype
        self.device = device
        match (len(coefs.shape), len(basis.shape)):
            case (1, 3):
                assert (coefs.shape[0] == basis.shape[0]), RuntimeError('coefs.shape[0] == basis.shape[0] ')
            case (1, 2):
                assert coefs.shape[0] == 1, RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
                basis = basis.reshape([1, *basis.shape])
            case (0, 2):
                basis = basis.reshape([1, *basis.shape])
                coefs = coefs.reshape([1])
            case (0,3):
                assert basis.shape[0] ==1, RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
                coefs = coefs.reshape([1])
            case _:
                raise RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
        self.basis:torch.Tensor = basis.to(dtype = dtype, device = device)
        self.coefs:torch.Tensor = coefs.to(dtype = dtype, device = device)
        self.is_zero = is_zero
        return
    
    def __matmul__(self, b:Self|State) -> Self:
        match b:
            case b if isinstance(b,PauliMatrix):
                if(b.is_zero):
                    return b
                elif(self.is_zero):
                    return self
                else:
                    A = doMul(
                            self.basis, self.coefs, b.basis, b.coefs
                        )
                    if(A is None):
                        return PauliMatrix(
                            torch.zeros((1,1,4), dtype=self.dtype),
                            torch.zeros((1), device=self.device),
                            is_zero= True,
                            dtype=self.dtype,
                            device=self.device
                        )
                    else:
                        return PauliMatrix(*A, is_zero=False, dtype=self.dtype, device = self.device)
            case b if isinstance(b,PauliState):
                if(self.is_zero):
                    return PauliState(torch.zeros_like(b.basis), b.objTp)
                else:
                    assert (b.objTp == 'ket'), 'To act on a state from the right must be ket'
                    return PauliState(
                        doMulStateMat(
                            self.basis, self.coefs, b.basis, b.coefs
                        ),
                        objTp='ket',
                        device=self.device,
                        dtype=self.dtype
                    )
            case _:
                raise TypeError('Must Be PauliState or PauliMatrix')

    def __rmatmul__(self, b:Self|State) -> Self:
        match b:
            case b if isinstance(b,PauliMatrix):
                if(b.is_zero):
                    return b
                elif(self.is_zero):
                    return self
                else:
                    return PauliMatrix(
                        *doMul(
                            self.basis, self.coefs, b.basis, b.coefs
                        )
                    )
            case b if isinstance(b,PauliState):
                if(self.is_zero):
                    return PauliState(torch.zeros(b.basis), b.objTp)
                else:
                    assert (b.objTp == 'bra'), 'To act on a state from the right must be bra'
                    return PauliState(
                        doMulStateMat(
                            self.basis, self.coefs, b.basis, b.coefs
                        ),
                        objTp='bra'
                    )
            case _:
                raise TypeError('Must Be PauliState or PauliMatrix but got {tp}'.format(tp=str(type(b))))

    def __add__(self, b:Self) -> Self:
        self.type_check(b)
        if(b.is_zero):
            return self
        elif(self.is_zero):
            return b
        A = addFun(self.basis, self.coefs, b.basis, b.coefs) 
        if(A is None):
            return PauliMatrix(
                torch.zeros((1, self.basis.shape[-2], 4), dtype = self.basis.dtype, device = self.basis.device), 
                torch.tensor([0.0], dtype=self.coefs.dtype, device = self.coefs.device), 
                is_zero=True
            )
        else:
            return PauliMatrix( 
                * A, is_zero=False
            )
    
    def __sub__(self, b:Self) -> Self:
        self.type_check(b)
        if(b.is_zero):
            return self
        elif(self.is_zero):
            return -1*b
        A = subFun(self.basis, self.coefs, b.basis, b.coefs) 
        if(A is None):
            return PauliMatrix(
                torch.zeros((1, self.basis.shape[-2], 4), dtype = self.basis.dtype, device = self.basis.device), 
                torch.tensor([0.0], dtype=self.coefs.dtype, device = self.coefs.device), 
                is_zero=True
            )
        else:
            return PauliMatrix( 
                *A
            )
    
    def __radd__(self, b:Self) -> Self:
        return self.__add__(b)
    
    def __rsub__(self, b:Self) -> Self:
        self.type_check(b)
        if(b.is_zero):
            return -1*self
        elif(self.is_zero):
            return b
        A = subFun(b.basis, b.coefs, self.basis, self.coefs) 
        if(A is None):
            return PauliMatrix(
                torch.zeros((1, self.basis.shape[-2], 4), dtype = self.basis.dtype, device = self.basis.device), 
                torch.tensor([0.0], dtype=self.coefs.dtype, device = self.coefs.device), 
                is_zero=True
            )
        else:
            return PauliMatrix( 
                *A
            )
    
    def __mul__(self, b:Self|torch.Tensor) -> Self:
        try:
            return self.__matmul__(b)
        except:
            assert (isinstance(b,torch.Tensor)), TypeError(f'Must be tensor type')
            assert (b.shape[0] == self.coefs.shape[0]), TypeError(f'Must be a Pauli Matrix or Tensor of size ({self.coefs.shape[0]},)')
            return PauliMatrix(basis=self.basis, coefs=self.coefs*b)
    
    def __rmul__(self, b:Self) -> Self:
        try:
            return self.__rmatmul__(b)
        except:
            assert (isinstance(b,torch.Tensor)), TypeError(f'Must be tensor type')
            assert (b.shape[0] == self.coefs.shape[0]), TypeError(f'Must be a Pauli Matrix or Tensor of size ({self.coefs.shape[0]},)')
            return PauliMatrix(basis=self.basis, coefs=self.coefs*b)
    
    def __add__(self, b:Self) -> Self:
        self.type_check(b)
        A = addFun(self.basis, self.coefs, b.basis, b.coefs) 
        if(A is None):
            return None
        else:
            return PauliMatrix( 
                *A
            )
    
    def __repr__(self) -> str:
        return f"PauliMatrix(basis = \n{self.basis.__repr__()},\ncoefs = \n{self.coefs.__repr__()},\nshape={self.basis.shape[:-2]}, is_zero={self.is_zero})"
    
    def __getitem__(self, ix:int|Iterable[int])->Self:
        assert (self.coefs.shape[0] != 1 ), IndexError('Pauli Matrix of with one basis is not indexed')
        return PauliMatrix(self.basis[ix], coefs=self.coefs[ix])
    
    def to_pauli_strings(self, to_ = 'PdDataFrame') -> PdDataFrame|PlDataFrame|PlLazyFrame|np.ndarray[str]|PyArrowTable|PlSeries|dict[str:PlSeries]|list[dict[str:str]]|str:
        def K(X, N):
            X[0] = X[0]//N
            X[1] = X[1]-N*X[0]
            return X
        e = toR(self.basis.flatten(end_dim=1)).flatten(start_dim = -2)
        v = torch.arange((e.shape[0])).reshape(e.shape[0], 1)
        e = torch.cat((v, v, e) , dim=1)
        names = ['BasisDir','HilbertSpace']
        opers = []
        for j in iter(['I', 'X', 'Y', 'Z']):
            names.extend([f'{j}Real', f'{j}Complex'])
            opers.append(
                when(
                    ((col(f'{j}Real')!= 0) | (col(f'{j}Complex') != 0))
                ).then(
                    when(
                        ((col(f"{j}Real")!=0) & (col(f'{j}Complex') == 0))
                    ).then(
                        plform("{}s", col(f"{j}Real").round_sig_figs(3))+f' {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")!=0) & (col(f'{j}Complex') > 0))
                    ).then(
                        '('+plform("{}", col(f"{j}Real").round_sig_figs(3))+'+'+plform("{}", col(f"{j}Complex").abs().round_sig_figs(3))+f'i) {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")!=0) & (col(f'{j}Complex') < 0))
                    ).then(
                        '('+plform("{}", col(f"{j}Real").round_sig_figs(3))+'-'+plform("{}", col(f"{j}Complex").abs().round_sig_figs(3))+f'i) {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")==0) & (col(f'{j}Complex') != 0))
                    ).then(
                        plform("{}", col(f"{j}Complex").round_sig_figs(3))+f'i {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    )
                ).alias(j)
            )
        Data = pl_from_numpy(
                    torch.vmap(
                        lambda x: K(x, self.basis.shape[1]), 
                        in_dims=0
                    )(e).numpy(), 
                    schema = names, 
                    schema_overrides={'BasisDir':Int32, 'HilbertSpace':Int32}
                ).lazy().with_columns(
                    opers
                ).select(
                    ('BasisDir', 'I','X','Y','Z')
                ).with_columns(
                    when(
                        ((col('I').is_null().not_()) & (col('X').is_null()) & (col('Y').is_null()) & (col('Z').is_null()))
                    ).then(
                        None
                    ).otherwise(
                        concat_str(
                            ('I','X','Y','Z'),
                            separator = '+', 
                            ignore_nulls=True
                        )
                    ).alias('Values')
                ).group_by('BasisDir').agg(
                    col('Values').str.concat(')\\otimes(', ignore_nulls=True)
                ).select('Values').with_columns(
                    (lit('$$(')+col('Values')+lit(")$$")).alias('Values')
                )
        
        match to_.lower():
            case 'pddataframe':
                return Data.collect().to_pandas()
            case 'pldataframe':
                return Data.collect()
            case 'pllazyframe':
                return Data
            case 'array':
                return Data.collect().to_numpy()
            case 'str':
                return ' + '.join(Data.collect().to_numpy()[:,0])
            case 'pyarrowtable':
                return Data.collect().to_arrow()
            case 'dict':
                return Data.collect().to_dict()
            case 'dicts':
                return Data.collect().to_dicts()
            case 'plseries':
                return Data.collect().to_series()    
            case _:
                warn('Returning Lazy Frame to_ should be in {PdDataFrame,PlDataFrame,PlLazyFrame, array, str, PyArrowTable, dict,dicts,PlSeries}')
                return Data
        
    def remove_zeros(self) -> None:
        c = toR(self.coefs).to(torch.bool).any(dim=-1)
        if(not c.all()):
            self.coefs = self.coefs[c]
            self.basis = self.basis[c]
        b = toR(self.basis).to(torch.bool).any(dim=-1).any(dim=-1).all(dim=-1)
        if(not b.all()):
            self.coefs = self.coefs[b]
            self.basis = self.basis[b]
        return
    
    def Tr(self, keep:torch.Tensor|Iterable[int]|None = None, tr_out:torch.Tensor|Iterable[int]|None = None) -> torch.Tensor|float:
        if(keep is None and tr_out is None ): 
            b = (torch.ceil(self.basis[...,0]) == 1).all(dim=1).to(self.basis.dtype)
            return fullTrace(self.basis, self.coefs)
        else:
            if(keep is not None):
                if(isinstance(keep, torch.Tensor)):
                    keep = keep.numpy()
                if(isinstance(keep, slice) or isinstance(keep, range)):
                    keep = np.arange(keep.start, keep.stop, keep.step)
                else:
                    keep = np.array(keep)
                tr_out = np.arange(self.basis.shape[0])
                tr_out = np.delete(tr_out, keep)
            else:
                if(isinstance(tr_out, torch.Tensor)):
                    tr_out= tr_out.numpy()
                if(isinstance(tr_out, slice) or isinstance(tr_out, range)):
                    tr_out= np.arange(tr_out.start, tr_out.stop, tr_out.step)
                else:
                    tr_out = np.array(tr_out)
                keep = np.arange(self.basis.shape[0])
                keep = np.delete(keep, tr_out)
            return PauliMatrix(ptrace(self.basis, self.coefs, torch.from_numpy(keep), torch.from_numpy(tr_out)))
    
    def dag(self, inplace:bool = True) -> None|Self:
        if(inplace):
            self.coefs.conj()
            self.basis.conj()
            return 
        else:
            return PauliMatrix(self.basis.clone().conj(), self.coefs.clone().conj())
        
    def transpose(self, inplace:bool = True) -> Self|None:
        if(inplace):
            self.basis[...,2]*=-1
            return
        else:
            b = self.basis.clone()
            b[...,2]*=-1
            return PauliMatrix(b, self.coefs.clone())
    
    def type_check(self, b:Self) -> bool:
        assert (isinstance(b,PauliMatrix)), TypeError('Must be PauliMatrix')
        return True
    
    def to(self,*args, inplace:bool = True, **kwargs) -> None|Self:
        basis = self.basis.to(*args,**kwargs)
        coefs = self.coefs.to(*args,**kwargs)
        if(inplace):
            self.basis = basis
            self.coefs = coefs
            if('device' in kwargs):
                self.device = kwargs['device']
            if('dtype' in kwargs):
                self.dtype = kwargs['dtype']
        else:
            return PauliMatrix(basis,coefs=coefs, is_zero= self.is_zero, dtype=self.dtype, device=self.device)
     
    def clone(self) -> Self:
        return PauliState(self.basis.clone(), self.coefs.clone(), dtype=self.dtype, device=self.device)

class PauliState(State):
    def __init__(
        self, basis:torch.Tensor, coefs:torch.Tensor, 
        time_dependence:Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        spatial_projection:Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        objTp:str = 'ket',
        dtype:torch.dtype= torch.complex128, 
        device:torch.device='cpu'
    ) -> Self:
        super(State,self).__init__()
        self.set_obj_tp(objTp)
        assert  isinstance(coefs,torch.Tensor) and isinstance(basis, torch.Tensor), TypeError('Must be Tensor type ')
        self.dtype = dtype
        self.device = device
        match (len(coefs.shape), len(basis.shape)):
            case (1, 3):
                assert (coefs.shape[0] == basis.shape[0]), RuntimeError('coefs.shape[0] == basis.shape[0] ')
            case (1, 2):
                assert coefs.shape[0] == 1, RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
                basis = basis.reshape([1, *basis.shape])
            case (0, 2):
                basis = basis.reshape([1, *basis.shape])
                coefs = coefs.reshape([1])
            case (0,3):
                assert basis.shape[0] ==1, RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
                coefs = coefs.reshape([1])
            case _:
                raise RuntimeError('Incompatible shapes for basis.shape = {base}, coefs.shape = {coef}'.format(base = basis.shape, coef=coefs.shape))
        
        self.basis:torch.Tensor = basis.to(dtype = dtype, device = device)
        self.coefs:torch.Tensor = coefs.to(dtype= dtype, device = device)
        return
    
    def normalize(self, inplace:bool = True) -> None|Self:
        N = ((self.basis * self.basis.conj()).sum(dim = -1).prod(dim = -1) @ (self.coefs.conj() * self.coefs)).sqrt()
        if(inplace):
            self.coefs /= N
            return 
        else:
            return PauliState(self.basis, self.coefs/N)
    
    def __mul__(self, B:Self|torch.Tensor|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B,torch.Tensor):
                return PauliState(self.basis*B, objTp=self.objTp)
            case B if isinstance(B, PauliMatrix) or isinstance(B, PauliState):
                return self.__matmul__(B)
            case _:
                raise TypeError('Must be Pauli Matrix Pauli State or Tensor')
                
    def __rmul__(self, B:Self|torch.Tensor|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B,torch.Tensor):
                return PauliState(B*self.basis, objTp=self.objTp)
            case B if isinstance(B, PauliMatrix) or isinstance(B, PauliState):
                return self.__rmatmul__(B)
            case _:
                raise TypeError('Must be Pauli Matrix Pauli State or Tensor')
                
    def __add__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        raise NotImplementedError('Not Yet implemented for PauliState + PauliState')
        return PauliState(self.basis+B.basis, objTp= self.objTp)
    
    def __radd__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        raise NotImplementedError('Not Yet implemented for PauliState + PauliState')
        return PauliState(self.basis+B.basis, objTp= self.objTp)
    
    def __matmul__(self, B:Self|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B, PauliMatrix):
                return B.__rmatmul__(self)
            case B if isinstance(B, PauliState):
                return doMulStateState(B.basis, B.objTp, self.basis, self.objTp)
            case _:
                raise TypeError('Must be PauliMatrix or PauliState')
    
    def __rmatmul__(self, B:Self|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B, PauliMatrix):
                return B.__matmul__(self)
            case B if isinstance(B, PauliState):
                return doMulStateState(B.basis, B.objTp, self.basis, self.objTp)
            case _:
                raise TypeError('Must be PauliMatrix or PauliState')
    
    def __sub__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        raise NotImplementedError('Not Yet implemented for PauliState + PauliState')
        return PauliState(self.basis-B.basis, objTp=self.objTp) 
    
    def __rsub__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        raise NotImplementedError('Not Yet implemented for PauliState + PauliState')
        return PauliState(B.basis-self.basis, objTp=self.objTp) 
    
    def __div__(self, B:torch.Tensor) -> Self:
        assert isinstance(B,torch.Tensor), TypeError('Must be Tensor type, Scalar or matching dims')
        return PauliState(self.basis/B, objTp=self.objTp)

    def dag(self, inplace:bool = False) -> Self|None:
        if(inplace):
            self.basis.conj()
            self.objTp = {'ket':'bra', 'bra':'ket'}[self.objTp]
            return  
        else:
            return PauliState(self.basis.conj(), objTp= {'ket':'bra', 'bra':'ket'}[self.objTp]) 
    
    def clone(self) -> Self:
        return  PauliState(self.basis.clone(), self.coefs.clone(), objTp=self.objTp, dtype=self.dtype, device=self.device)
    
    def to(self,*args, inplace:bool = True, **kwargs) -> None|Self:
        basis = self.basis.to(*args,**kwargs)
        coefs = self.coefs.to(*args,**kwargs)
        if(inplace):
            self.basis = basis
            self.coefs = coefs
            if('device' in kwargs):
                self.device = kwargs['device']
            if('dtype' in kwargs):
                self.dtype = kwargs['dtype']
        else:
            return PauliState(basis,coefs=coefs, objTp=self.objTp, dtype=self.dtype, device=self.device)
    
    def __repr__(self,)->str:
        return f"\\{self.objTp}{{\\psi}} = PauliState(basis = \n{self.basis.__repr__()},\ncoefs = \n{self.coefs.__repr__()},\nshape={self.basis.shape[:-2]})"
