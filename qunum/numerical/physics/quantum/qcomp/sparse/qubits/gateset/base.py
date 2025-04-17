from typing import Iterable, Iterator
import torch
from .....qobjs.sparse_su2_dep import SU2Matrix
class Gate:
    def __init__(self, n_qubits:int, target:int|Iterable[int], device:torch.device = 'cpu', dtype:torch.dtype=torch.complex128, niter:int = 1,  **kwargs):
        self.n_qubits =n_qubits,
        self.target = target
        self.device = device
        self.dtype = dtype
        self.niter = niter
        self.n = 0
        return
    
    def __call__(self)->SU2Matrix:
        pass 

    def __next__(self)->SU2Matrix:
        pass
        
    def __iter__(self)->Iterator[SU2Matrix]:
        pass

