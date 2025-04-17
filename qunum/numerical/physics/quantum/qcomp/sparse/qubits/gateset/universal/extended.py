from ......qobjs.sparse_su2_dep import SU2Matrix
import torch
from typing import Generator, Iterator, Iterable
from ..base import Gate

class TGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        return 
    
    def __call__(self)->SU2Matrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 1] = 1
        Basis[0, self.target] = torch.tensor([1+1j,0, 0, 1-1j], dtype=self.dtype, device=self.device)
        return SU2Matrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    
    def __next__(self)->SU2Matrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    def __iter__(self)->Iterator[SU2Matrix]:
        return self
    

class PhaseGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], theta:torch.Tensor|float, *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        if not (isinstance(theta,torch.Tensor)):
            self.theta = torch.tensor(theta)
        else:
            self.theta = theta
        return
    
    def __call__(self)->SU2Matrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        Basis[0, self.target] = torch.tensor([1+torch.exp(1j*self.theta/2),0, 0, 1-torch.exp(1j*self.theta/2)], dtype=self.dtype, device=self.device)
        return SU2Matrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    
    def __next__(self)->SU2Matrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    
    def __iter__(self)->Iterator[SU2Matrix]:
        return self

class SwapGate(Gate):    
    def __init__(self, n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        assert len(target) == 2, TypeError('Must be a two qubit swap ')
        super().__init__(n_qubits, target, device, dtype, niter)
        return
    
    def __call__(self)->SU2Matrix:
        Basis = torch.zeros((4,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        e = torch.eye(4, dtype=self.dtype, device = self.device)
        for i in range(4):
            Basis[i, self.target] = e[[i,i]] 
        return SU2Matrix(Basis, coefs=torch.ones(4, dtype = self.dtype, device= self.device)/2)
    
    def __next__(self)->SU2Matrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    
    def __iter__(self)->Iterator[SU2Matrix]:
        return self


def t_gate(n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[SU2Matrix, None, None]:
    B = torch.tensor([1 + torch.exp(1j* torch.tensor(torch.pi)/4),0, 0, 1 - torch.exp(1j* torch.tensor(torch.pi)/4)])
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 1] = 1
        Basis[0, target] = B
        yield SU2Matrix(Basis, coefs = torch.tensor([1+0j], dtype = dtype, device = device))
        if(stpIt):
            break

def phase_gate(n_qubits:int, target:int|Iterable[int], theta:torch.Tensor, *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[SU2Matrix, None, None]:
    B = torch.tensor([1 + torch.exp(1j* theta/2),0, 0, 1 - torch.exp(1j* theta/2)])
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 1] = 1
        Basis[0, target] = B
        yield SU2Matrix(Basis, coefs = torch.tensor([1+0j], dtype = dtype, device = device))
        if(stpIt):
            break