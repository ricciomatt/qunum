import torch 
import numpy as np
from .core import addFun, subFun, doMul, ptrace, fullTrace, doMulStateMat, doMulStateState
from typing import Self, Iterable
from torch import view_as_real as toR, view_as_complex as toC
from polars import from_numpy as pl_from_numpy, concat_str, col, when, lit, Int32, String, DataFrame as PlDataFrame, LazyFrame as PlLazyFrame, Series as PlSeries
from pandas import DataFrame as PdDataFrame
from pyarrow import Table as PyArrowTable
from warnings import warn

class State:
    def __init__(self, objTp:str = 'ket') -> Self:
        self.objTp:str= objTp
        return


class PauliMatrix:
    def __init__(self, basis:torch.Tensor, coefs:torch.Tensor, dtype = torch.complex128)->Self:
        assert isinstance(basis, torch.Tensor), TypeError('Must be Tensor type ')
        assert isinstance(coefs, torch.Tensor), TypeError('Must be Tensor type')
        assert basis.shape[-1] == 4, RuntimeError('basis.shape[-1] == 4')
        if(len(coefs.shape) == 0):
            assert len(basis.shape) == 2 or (len(basis.shape) == 3 and basis.shape[0] == 1), RuntimeError('Error a scalar coefficent requires scalar coeficents')
            coefs = coefs.reshape([1])
            match basis.shape:
                case shp if len(shp) == 2:
                    basis = basis.reshape((1,*basis.shape))
                case _:
                    basis = basis.reshape((1, *basis.shape[1:]))
        else:
            assert (coefs.shape[0] == basis.shape[0]), RuntimeError('coefs.shape[0] == basis.shape[0] ')
        self.basis:torch.Tensor = basis.to(dtype = dtype)
        self.coefs:torch.Tensor = coefs.to(dtype= dtype)
        return
    
    def __matmul__(self, b:Self|State) -> Self:
        match b:
            case b if isinstance(b,PauliMatrix):
                return PauliMatrix(
                    *doMul(
                        self.basis, self.coefs, b.basis, b.coefs
                    )
                )
            case b if isinstance(b,PauliState):
                return PauliState(
                    doMulStateMat(
                        self.basis, self.coefs, b.basis
                    )
                )
            case _:
                raise TypeError('Must Be PauliState or PauliMatrix')

    def __add__(self, b:Self) -> Self:
        self.type_check(b)
        return PauliMatrix( 
            *addFun(self.basis, self.coefs, b.basis, b.coefs) 
        )
    
    def __sub__(self, b:Self) -> Self:
        self.type_check(b)
        A = subFun(self.basis, self.coefs, b.basis, b.coefs) 
        if(A is None):
            pass
        else:
            return PauliMatrix( 
                *A
            )
    
    def __radd__(self, b:Self) -> Self:
        return self.__add__(b)
    
    def __rsub__(self, b:Self) -> Self:
        self.type_check(b)
        A = subFun(b.basis, b.coefs, self.basis, self.coefs) 
        if(A is None):
            A = torch.zeros((self.basis.shape[0],1,4), dtype=torch.complex128)
            A[:,:,0] = 1
            return PauliMatrix(
                A,
                torch.zeros(1)
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
    
    def type_check(self, b:Self) -> bool:
        assert (isinstance(b,PauliMatrix)), TypeError('Must be PauliMatrix')
        return True
    
    def __rmatmul__(self, b:Self|State) -> Self:
        match b:
            case b if isinstance(b,PauliMatrix):
                return PauliMatrix(
                    *doMul(
                        self.basis, self.coefs, b.basis, b.coefs
                    )
                )
            case b if isinstance(b,PauliState):
                return PauliState(
                    doMulStateMat(
                        self.basis, self.coefs, b.basis
                    )
                )
            case _:
                raise TypeError('Must Be PauliState or PauliMatrix')

    def __add__(self, b:Self) -> Self:
        self.type_check(b)
        return PauliMatrix( 
            *addFun(self.basis, self.coefs, b.basis, b.coefs) 
        )
    
    def __repr__(self) -> str:
        return f"PauliMatrix(basis = \n{self.basis.__repr__()}\n coefs = {self.coefs.__repr__()})"
    
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
                        col(f"{j}Real").cast(String)+f' {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")!=0) & (col(f'{j}Complex') > 0))
                    ).then(
                        '('+col(f"{j}Real").cast(String)+'+'+col(f'{j}Complex').cast(String)+f'i) {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")!=0) & (col(f'{j}Complex') < 0))
                    ).then(
                        '('+col(f"{j}Real").cast(String)+'-'+col(f'{j}Complex').abs().cast(String)+f'i) {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
                    ).when(
                        ((col(f"{j}Real")==0) & (col(f'{j}Complex') != 0))
                    ).then(
                        col(f'{j}Complex').abs().cast(String)+f'i {j}_'+'{'+col('HilbertSpace').cast(String)+'}'
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
    
    def Diagonalize(self, inplace:bool = False) -> None|Self:
        pass
    
    def getU(self, inplace:bool = False) -> None|Self:
        pass
   
    def expm(self) -> Self:
        return




class PauliState(State):
    def __init__(self, basis:torch.Tensor, objTp:str = 'ket') -> Self:
        super(State,self).__init__()
        assert isinstance(basis, torch.Tensor) , TypeError('Basis Must be a torch Tensor with and basis.shape[1] = 2 and len(basis.shape) == 2')
        self.basis:torch.Tensor = basis
        self.objTp = objTp
        return
    
    def normalize(self, inplace:bool = True) -> None|Self:
        N = (self.basis * self.basis.conj()).sum().sqrt()
        if(inplace):
            self.basis /= N
            return
        else:
            return PauliState(self.basis/N)
    
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
        return PauliState(self.basis+B.basis, objTp= self.objTp)
    
    def __radd__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        return PauliState(self.basis+B.basis, objTp= self.objTp)
    
    def __matmul__(self, B:Self|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B, PauliMatrix):
                return PauliState(doMulStateMat(B.basis, B.coefs, self.basis), objTp=self.objTp)
            case B if isinstance(B, PauliState):
                return doMulStateState(B.basis, B.objTp, self.basis, self.objTp)
            case _:
                raise TypeError('Must be PauliMatrix or PauliState')
    
    def __rmatmul__(self, B:Self|PauliMatrix) -> Self|torch.Tensor:
        match B:
            case B if isinstance(B, PauliMatrix):
                return PauliState(doMulStateMat(B.basis, B.coefs, self.basis), objTp=self.objTp)
            case B if isinstance(B, PauliState):
                return doMulStateState(B.basis, B.objTp, self.basis, self.objTp)
            case _:
                raise TypeError('Must be PauliMatrix or PauliState')
    
    def __sub__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
        return PauliState(self.basis-B.basis, objTp=self.objTp) 
    
    def __rsub__(self, B:Self) -> Self:
        assert isinstance(B,PauliState), TypeError('Must be PauliState Object')
        assert B.objTp == self.objTp,  TypeError(f'Must be PauliState objTp should should match got {self.objTp} != {B.objTp}')
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
    
        

