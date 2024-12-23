from typing import Iterator, Callable
from .base import Gate
from ....qobjs.sparse_su2 import PauliMatrix
import torch
class CustomOperator(Gate):
    def __init__(self, operator:PauliMatrix, device:torch.device='cpu', dtype:torch.dtype = torch.complex128, **kwargs):
        super().__init__(operator.basis.shape[1], target = None, device=device, dtype = dtype, **kwargs)
        self.operator = operator
        self.operator.to(dtype=dtype, device = device)
        return
    def __call__(self) -> PauliMatrix:
        return self.operator
    def __iter__(self) -> Iterator[PauliMatrix]:
        return self
    def __next__(self) -> PauliMatrix:
        if(self.n<self.niter):
            return self()
        else:
            self.n = 0 
            raise StopIteration
    
class CustomCallableOperator:
    def __init__(self, operator:Callable[[tuple,dict],PauliMatrix], n_qubits:int, *args, device:torch.device='cpu', dtype:torch.dtype = torch.complex128, **kwargs):
        self.operator = operator
        self.device = device
        self.dtype = dtype
        self.args = args
        self.kwargs = kwargs
        self.n_qubits = n_qubits
        return
    def __call__(self):
        op  = self.operator(*self.args, **self.kwargs)
        op.to(dtype=self.dtype, device = self.device)
        return op
    
    def __iter__(self) -> Iterator[PauliMatrix]:
        return self
    def __next__(self) -> PauliMatrix:
        if(self.n<self.niter):
            return self()
        else:
            self.n = 0 
            raise StopIteration