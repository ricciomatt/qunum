import torch 
import numpy as np
from .core import addFun, subFun, doMul, ptrace, fullTrace
from typing import Self, Iterable
from torch import view_as_real as toR, view_as_complex as toC


class PauliMatrix:
    def __init__(self, basis:torch.Tensor, coefs:torch.Tensor)->Self:
        assert isinstance(basis, torch.Tensor), TypeError('Must be Tensor type ')
        assert isinstance(coefs, torch.Tensor), TypeError('Must be Tensor type')
        assert basis.shape[-1] == 4, RuntimeError('basis.shape[-1] == 4')
        assert (coefs.shape[0] == basis.shape[0]), ValueError('coefs.shape[0] == basis.shape[0] ')
        self.basis:torch.Tensor = basis
        self.coefs:torch.Tensor = coefs
        return
    
    def __matmul__(self, b:Self)->tuple[Self,torch.Tensor]:
        self.type_check(b)
        return PauliMatrix(
            *doMul(
                self.basis, self.coefs, b.basis, b.coefs
            )
        )
    
    def __add__(self, b:Self)->tuple[Self, torch.Tensor]:
        self.type_check(b)
        return PauliMatrix( 
            *addFun(self,b) 
        )
    
    def __sub__(self, b:Self)->Self:
        self.type_check(b)
        return PauliMatrix( 
            *subFun(self,b) 
        )
    
    def __radd__(self, b:Self)->Self:
        return self.__add__(b)
    
    def __rsub__(self, b:Self)->Self:
        return self.__sub__(b)
    
    def __mul__(self, b:Self|torch.Tensor)->Self:
        try:
            return self.__matmul__(b)
        except:
            assert (isinstance(b,torch.Tensor)), TypeError(f'Must be tensor type')
            assert (b.shape[0] == self.coefs.shape[0]), TypeError(f'Must be a Pauli Matrix or Tensor of size ({self.coefs.shape[0]},)')
            return PauliMatrix(basis=self.basis, coefs=self.coefs*b)
    
    def __rmul__(self, b:Self)->Self:
        try:
            return self.__rmatmul__(b)
        except:
            assert (isinstance(b,torch.Tensor)), TypeError(f'Must be tensor type')
            assert (b.shape[0] == self.coefs.shape[0]), TypeError(f'Must be a Pauli Matrix or Tensor of size ({self.coefs.shape[0]},)')
            return PauliMatrix(basis=self.basis, coefs=self.coefs*b)
   
    def __rmatmul__(self, b:Self)->Self:
        self.type_check(b)
        return PauliMatrix(
            *doMul(
                b.basis, b.coefs, self.basis, self.coefs
            )
        )
    
    def type_check(self, b:Self)->bool:
        assert(isinstance(b,PauliMatrix)), TypeError('Must be a PauliMatrix')
        return True
    
    def __repr__(self)->str:
        return f"PauliMatrix(basis = \n{self.basis.__repr__()}\n coefs = {self.coefs.__repr__()})"
    
    def remove_zeros(self)->None:
        c = toR(self.coefs).to(torch.bool).any(dim=-1)
        if(not c.all()):
            self.coefs = self.coefs[c]
            self.basis = self.basis[c]
        b = toR(self.basis).to(torch.bool).any(dim=-1).any(dim=-1).all(dim=-1)
        if(not b.all()):
            self.coefs = self.coefs[b]
            self.basis = self.basis[b]
        return
    
    def Tr(self, keep:torch.Tensor|Iterable[int]|None = None, tr_out:torch.Tensor|Iterable[int]|None = None)->torch.Tensor|float:
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
    
    def Diagonalize(self, inplace:bool = False)->None|Self:
        pass
    
    def getU(self, inplace:bool = False)->None|Self:
        pass

    def dag(self, inplace:bool = True)->None|Self:
        if(inplace):
            self.coefs.conj()
            self.basis.conj()
            return 
        else:
            return PauliMatrix(self.basis.clone().conj(), self.coefs.clone().conj())
        
    def transpose(self, inplace:bool = True)->Self|None:
        if(inplace):
            self.basis[...,2]*=-1
            return
        else:
            b = self.basis.clone()
            b[...,2]*=-1
            return PauliMatrix(b, self.coefs.clone())

class PauliState:
    def __init__(self, data:torch.Tensor)->None:
        return
        

