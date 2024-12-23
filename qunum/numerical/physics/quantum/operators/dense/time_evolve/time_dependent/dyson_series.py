from .....qobjs import TQobj, eye
import torch 
from typing import Callable, Iterable, Iterator, Any
from .......mathematics.numerics.integrators.newton import NewtonCoates, integrate_newton_coates_do as newton
from IPython.display import display as disp, Markdown as md 
import numpy as np

class DysonSeriesGenerator:
    def __init__(
            self, 
            Hamiltonian:Callable[[torch.Tensor], TQobj], 
            order:int=int(2), 
            Integrator:NewtonCoates = NewtonCoates(2, dtype = torch.complex128), 
            dt:float = float(1e-3), 
            niter_eval:int = int(5e1), 
            num_pts:int = int(1e3),
            hbar:float = float(1.0),
            **kwargs:dict[str:Any,],
        )->object:
        assert callable(Hamiltonian), TypeError('Hamiltonian Must be a callable and take in a time dimension')
        assert isinstance(order, int), TypeError('Order must be an integer value')
        self.Hamiltonian = Hamiltonian
        self.dt = float(dt)
        self.order = int(order)
        self.niter_eval = niter_eval
        self.num_pts = num_pts
        self.IntegratorCoefficents = Integrator.L
        self.n0 = 0
        self.n = 0
        self.hbar = float(hbar)
        return

    
    def evolve(self, a:float|torch.Tensor, b:float|torch.Tensor, num_pts:int|None = None)->TQobj:
        if(num_pts is None):
            num_pts = self.num_pts
        return self(torch.linspace(a, b, num_pts))
    
    def __call__(self, t:torch.Tensor)->TQobj:
        return self.getUSeries(t).sum(dim=0)
    
    def getUSeries(self, t:torch.Tensor)->TQobj:
        H = self.Hamiltonian(t)
        U = TQobj(dyson_series(H, self.dt, order=self.order, L = self.IntegratorCoefficents, hbar=self.hbar), meta = H._metadata)
        del H
        return U
    
    def __getitem__(self, ix:tuple[int,int]|torch.Tensor|Iterable[int]|int)->TQobj:
        if(isinstance(ix, tuple)):
            return self.evolve(ix[0], ix[1], self.num_pts)
        if(isinstance(ix, torch.Tensor)):
            assert (torch.Tensor.dtype not in [torch.int16, torch.int32, torch.int64, torch.int8]), 'Must be Integer Index'
            return self(ix*self.dt)
        if(isinstance(ix, list) or isinstance(ix, np.ndarray)):
            ix = np.array(ix).astype(np.int64)
            return self(torch.from_numpy(ix)*self.dt)
        elif(isinstance(ix,int)):
            return self(torch.tensor([ix])*self.dt)
        else:
            raise ValueError('Could Not resolve Item')
    
    def __iter__(self, t)->Iterator:
        return self
    
    def __next__(self)->TQobj:
        if(self.n<self.num_pts):
            t = torch.linspace(self.n*self.dt - self.dt/2, (self.n+1/2)*self.dt, self.niter_eval)
            self.n+=1
            return self(t)
        else:
            self.n0+=self.n
            self.n = 0
            raise StopIteration
    
    def reset_iter(self)->None:
        self.n0 = 0
        self.n = 0
        return
    
    def __repr__(self)->str:
        return f"DysonSeriesGenerator({str(self.Hamiltonian)} \ndt={self.dt},\norder={self.order},\n integrator=NewtonCoates({self.IntegratorCoefficents.shape[0]})"


@torch.jit.script
def dyson_series(H:torch.Tensor, dt:float, order:int, L:torch.Tensor, hbar:float)->torch.Tensor:
    H *= torch.complex(torch.tensor(0.0), torch.tensor(-1/hbar)).to(dtype=H.dtype, device = H.device)
    U = torch.empty((order, H.shape[0], H.shape[1], H.shape[2]), dtype = H.dtype, device = H.device)
    U[0,:] = torch.eye(H.shape[1], H.shape[2], dtype = H.dtype, device = H.device)
    for o in range(1, order):
        U[o] = newton(H@U[o-1], L)*dt
    return U