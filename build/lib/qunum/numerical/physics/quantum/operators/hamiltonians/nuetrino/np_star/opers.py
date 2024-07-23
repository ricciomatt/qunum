import torch
from .....qobjs import TQobj
from .......mathematics.algebra import representations as reprs
from .......mathematics.einsum import einsum as qein
from typing import Callable, Generator
from ....general_operators.ladder.jordan_wigner import jordan_wigner_su2

class TwoPHamiltonian:
    def __init__(self, N:int, p_states:torch.Tensor, p_to_w:Callable[[torch.Tensor], torch.Tensor], niter:int, dt:float = 1e-4, **kwargs:dict[str:torch.Tensor] )->None:
        assert p_states.shape[0] == 2, SyntaxError('Must be size 2 tensorfor p_states')
        self.N = N
        self.np = 2
        self.p_states = p_states
        self.wp = p_to_w(p_states)
        self.niter = niter
        self.a = self.getLadder()
        self.Hk:TQobj = (self.a.dag() @ self.a).sum(dim=0)
        self.Hvv:TQobj = (self.a.dag() @ self.a.dag() @ self.a @ self.a )
        self.n = 0
        self.dt = dt
        return
     
    def __iter__(self)->Generator[TQobj, None, None]:
        return self
    
    def getLadder(self, )->TQobj:
        return jordan_wigner_su2(self.N+self.np, {n:4 for n in range(self.N)})
         
    
    def __next__(self)->TQobj:
        if(self.n<self.niter):
            self.n+=1
            return torch
        else:
            raise StopIteration
        return
    
    def FlavortoMass(self, psi:TQobj)->TQobj:
        return 


def get_Hvv()->TQobj:
    return