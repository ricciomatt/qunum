from ......qobjs.sparse_su2_dep import SU2Matrix
import torch
from typing import Generator, Iterable, Self, Iterator
from ..base import Gate
from ........mathematics.combintorix import EnumerateArgCombos as enumIt
    
class ToffoliGate(Gate):
    def __init__(self, n_qubits:int, target:int|Iterable[int], controll:int|Iterable[int], *args:tuple, device:torch.device = 'cpu', dtype:torch.dtype = torch.complex128, niter:int= 1, **kwargs:dict):
        super().__init__(n_qubits, target, device, dtype, niter)
        assert len(target), ValueError('Must have 2 controll qubits')
        self.controll = controll
        dir:int = dict(x = 1, y = 2, z = 3)[dir]
        basis = torch.eye(4, dtype= dtype, device=device)
        enum = enumIt(torch.tensor([0,3]), torch.tensor([0,3]), torch.tensor([0,1]))()
        self.Basis = basis[enum]
        self.Coefs:torch.Tensor = torch.pow(torch.tensor(-1+0j, device = device), (enum%2).sum(dim=-1)+1).to(dtype = dtype)
        self.Coefs[0] = 3
        self.Coefs /= 4
        return 

    def __call__(self)->SU2Matrix:     
        Basis = torch.zeros((self.Basis.shape[0], self.n_qubits, 4), dtype = self.dtype, device = self.device)
        Basis[:, :, 0] = 1+0j
        Basis[:, self.controll] = self.Basis[:, [0,1]]
        Basis[:, self.target] = self.Basis[:, -1]
        return SU2Matrix(Basis, coefs=self.Coefs)
    
    def __next__(self)->SU2Matrix:
        if(self.n < self.niter):
            self.n+=1
            return self()
        else:
            self.n = 0
            raise StopIteration
    
    def __iter__(self)->Iterator[SU2Matrix]:
        return self
