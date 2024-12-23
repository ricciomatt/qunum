from typing import Iterable, Iterator
import torch
from ....qobjs import PauliMatrix
class Gate:
    def __init__(self, n_qubits:int, target:int|Iterable[int], device:torch.device = 'cpu', dtype:torch.dtype=torch.complex128, niter:int = 1,  **kwargs):
        self.n_qubits =n_qubits,
        self.target = target
        self.device = device
        self.dtype = dtype
        self.niter = niter
        self.n = 0
        return
    
    def __call__(self)->PauliMatrix:
        pass 

    def __next__(self)->PauliMatrix:
        pass
        
    def __iter__(self)->Iterator[PauliMatrix]:
        pass

