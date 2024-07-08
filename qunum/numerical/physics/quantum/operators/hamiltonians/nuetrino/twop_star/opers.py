import torch
from .....qobjs import TQobj
from .......mathematics.algebra import representations as reprs
from .......mathematics.einsum import einsum as qein


class NPHamiltonian:
    def __init__(self,N:int, p_states:torch.Tensor, niter:int, dt:float = 1e-4, **kwargs:dict[str:torch.Tensor] )->None:
        self.N = N
        self.p_states = p_states
        self.niter = niter
        self.n = 0
        self.dt = dt
        return
     
    def __iter__(self)->object:
        return self
    
    def __next__(self)->torch.Tensor:
        if(self.n<self.niter):
            self.n+=1
            return torch
        else:
            raise StopIteration
        return
