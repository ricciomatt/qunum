from .....qobjs.sparse_su2 import PauliMatrix
import torch
from typing import Generator, Iterable, Self, Iterator
from math import sqrt
from ..base import Gate
from .......mathematics.combintorix import EnumerateArgCombos as enumIt

class PauliGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], *args:tuple, dir_:str='x', device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        self.dir = dict(x = 1, y = 2, z = 3)[dir_]
        return 
    def __call__(self)->PauliMatrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        Basis[0, self.target] = torch.tensor([1+0j if(i == self.dir) else 0.0 + 0j for i in range(4)], dtype =self.dtype, device=self.device)
        return PauliMatrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    def __next__(self)->PauliMatrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    def __iter__(self)->Iterator[PauliMatrix]:
        return self

class RotationGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], theta:torch.Tensor|float, *args:tuple, dir_:str='x', device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        if not (isinstance(theta, torch.Tensor)):
            self.theta = torch.tensor(theta)
        else:
            self.theta = theta
        self.dir = dict(x = 1, y = 2, z = 3)[dir_]
        return 
    
    def __call__(self)->PauliMatrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        Basis[0, self.target] = torch.tensor([-1j*(self.theta/2).sin() if(i == self.dir) else (self.theta/2).cos() if(i ==0) else 0+0j for i in range(4)], dtype =self.dtype, device=self.device)
        return PauliMatrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    
    def __next__(self)->PauliMatrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
        
    def __iter__(self)->Iterator[PauliMatrix]:
        return self

class Hadamard(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        return 
    
    def __call__(self)->PauliMatrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        Basis[0, self.target] = torch.tensor([0,1/sqrt(2), 0, 1/sqrt(2)])
        return PauliMatrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    
    def __next__(self)->PauliMatrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    def __iter__(self)->Iterator[PauliMatrix]:
        return self

class SGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        return 
    
    def __call__(self)->PauliMatrix:
        Basis = torch.zeros((1,self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[0,:, 0] = 1
        Basis[0, self.target] = torch.tensor([1+1j,0, 0, 1-1j], dtype=self.dtype, device=self.device)
        return PauliMatrix(Basis, coefs=torch.tensor([1.0+0j], dtype = self.dtype, device= self.device))
    
    def __next__(self)->PauliMatrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    def __iter__(self)->Iterator[PauliMatrix]:
        return self
    
class ControlledPauli(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], controll:int|Iterable[int], pauli_dir:int|str = 1, *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        if not (isinstance(target, int)):
            raise NotImplemented('N Qubit controlled operation are not yet implemented')
        if not (isinstance(controll, int)):
            raise NotImplemented('N Qubit controlled operations are not yet implemented')
        if(isinstance(pauli_dir, int)):
            assert pauli_dir>0 and pauli_dir<4, 'Pauli dir Must be number between 1 and 3'
            dir_ = pauli_dir
        else:
            assert pauli_dir.lower() in {'x','y','z'}, 'Pauli dir Must be number between 1 and 3 or string in {"x","y","z"}'
            dir_:int = dict(x = 1, y = 2, z = 3)[pauli_dir]

        super().__init__(n_qubits, target, device, dtype, niter)
        self.controll = controll
        basis = torch.eye(4, dtype= dtype, device=device)
        #k = [torch.tensor([0,3]) if c<len(controll) else torch.tensor([0, dir]) for c in range(len(controll)+len(target))]
        
        enum = enumIt(torch.tensor([0,3]), torch.tensor([0,dir_]))()
        self.Basis:torch.Tensor = basis[enum]
        self.Coefs:torch.Tensor = torch.pow(torch.tensor(-1+0j, device = device), (enum%2).sum(dim=-1)+1).to(dtype = dtype)
        self.Coefs[0] = 1
        self.Coefs/= 2
        return 
    def __call__(self)->PauliMatrix:     
        Basis = torch.zeros((self.Basis.shape, self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[:, :, 0] = 1+0j
        Basis[:, self.controll] = self.Basis[:, [0,1]]
        Basis[:, self.target] = self.Basis[:, -1]
        return PauliMatrix(Basis, coefs=self.Coefs)
    
    def __next__(self)->PauliMatrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    
    def __iter__(self)->Iterator[PauliMatrix]:
        return self




def pauli(n_qubits:int, target:int|Iterable[int], *args:tuple, dir_:str='x', device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs:dict)->Generator[PauliMatrix, None, None]:
    dir_dict = dict(x = 1, y = 2, z = 3)
    B = torch.tensor([1+0j if(i == dir_dict[dir_]) else 0.0 + 0j for i in range(4)], dtype =dtype, device=device)
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 0] = 1
        Basis[0, target] = B
        yield PauliMatrix(Basis, coefs=torch.tensor([1.0+0j], dtype = dtype, device= device))
        if(stpIt):
            break

def rot( n_qubits:int, target:int|Iterable[int], theta:torch.Tensor|float, *args:tuple, dir_:str = 'x', device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[PauliMatrix, None, None]:
    if not (isinstance(theta, torch.Tensor)):
        theta = torch.tensor(theta)
    dir_dict = dict(x = 1, y = 2, z = 3)
    B = torch.tensor([-1j*(theta/2).sin() if(i == dir_dict[dir_]) else (theta/2).cos() if(i ==0) else 0+0j for i in range(4)], dtype =dtype, device=device)
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 0] = 1
        Basis[0, target] = B
        yield PauliMatrix(Basis, coefs = torch.tensor([1.0+0j], dtype = dtype, device = device))
        if(stpIt):
            break

def hadamard(n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[PauliMatrix, None, None]:
    B = torch.tensor([0,1/sqrt(2), 0, 1/sqrt(2)])
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 0] = 1
        Basis[0, target] = B
        yield PauliMatrix(Basis, coefs = torch.tensor([1+0j], dtype = dtype, device = device))
        if(stpIt):
            break

def s_gate(n_qubits:int, target:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[PauliMatrix, None, None]:
    B = torch.tensor([1+1j,0, 0, 1-1j])
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        Basis[0,:, 0] = 1
        Basis[0, target] = B
        yield PauliMatrix(Basis, coefs = torch.tensor([1+0j], dtype = dtype, device = device))
        if(stpIt):
            break

def controlled_pauli(n_qubits:int, target:int|Iterable[int], control:int|Iterable[int],  *args:tuple, dir_:str='x', device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[PauliMatrix, None, None]:
    dir_:int = dict(x = 1, y = 2, z = 3)[dir_]
    V = torch.zeros(4, dtype = dtype, device = device)
    V[dir_] = 1
    I = torch.tensor([1,0,0,0], dtype = dtype, device = device)
    Z = torch.tensor([0,0,0,1], dtype = dtype, device = device)
    TOp:list[torch.Tensor] = [I.clone(), V.clone(), I.clone(), V.clone()]
    COp:list[torch.Tensor] = [I.clone(), I.clone(), Z.clone(), Z.clone()]
    Coefs:torch.Tensor = torch.ones(4, device=device, dtype=dtype)/2.0
    Coefs[-1] *= -1
    while True:
        Basis = torch.zeros((4, n_qubits, 4), dtype = dtype, device = device)
        for i in range(4):
            Basis[i, control] = COp[i]
            Basis[i, target] = TOp[i]
        yield PauliMatrix(Basis, coefs = Coefs)
        if(stpIt):
            break

def controlled_rotation(target:int|Iterable[int], control:int|Iterable[int], n_qubits:int, theta:float|torch.Tensor, *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, stpIt:bool = True, **kwargs)->Generator[PauliMatrix, None, None]:
    while True:
        Basis = torch.zeros((1,n_qubits, 4), dtype = dtype, device = device)
        yield PauliMatrix(Basis, coefs = torch.tensor([1+0j], dtype = dtype, device = device))
        if(stpIt):
            break