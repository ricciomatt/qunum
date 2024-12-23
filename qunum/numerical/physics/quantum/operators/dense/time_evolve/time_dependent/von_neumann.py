
from .......mathematics.algebra import ad
from .....qobjs import TQobj
from typing import Self, Callable, Generator
from torch import Tensor, tensor 


class VonNeumannEquation:
    def __init__(self, KraussOpers:TQobj)->Self:
        pass
    def __call__(self):
        pass

class RecursiveEvolve:
    def __init__(self, H:Callable[[Tensor], TQobj], dt:float = 1e-3, hbar:float= 1.):
        self.H = H
        self.dt = tensor(dt)
        self.hbar = tensor(hbar)
        return 
    def __call___(self, rho0:TQobj, n:int = int(1e2))->TQobj:
        rho = -1j*self.dt/self.hbar*ad(self.H(n*self.dt), rho0)
        if(n>0):
            return self(rho, n-1)
        return rho

    def __iter__(self, rho:TQobj, n:int = int(1e2))->Generator[TQobj, None, None]:
        def evolveIt(H:Callable[[Tensor], TQobj], rho:TQobj, n:int, hbar:float, dt:Tensor)->Generator[TQobj, None, None]:
            i = 0
            for i in range(n):
                rho = ad(-1j*H(i*dt)*(dt/hbar), rho)
                yield rho
        if(isinstance(self.dt, Tensor)):
            self.dt = tensor(self.dt)
        if(isinstance(self.hbar, Tensor)):
            self.hbar = tensor(self.hbar)
        return evolveIt(self.H, rho, n, self.hbar, self.dt)