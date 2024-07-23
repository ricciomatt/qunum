from ....qobjs import TQobj, zeros_like
from typing import Self, Generator, Callable
import torch

class ManySuddenApprx:
    def __init__(self, H:Callable[[torch.Tensor], TQobj], num_pts:int = int(1e3), dt:float = 1e-3, niter_eval:int=int(50))->Self:
        self.H = H
        self.niter_eval = niter_eval
        self.num_pts = num_pts
        self.dt = dt 
        return 
    
    def __iter__(self)->Generator[TQobj, None, None]:
        def yield_time_evolve(self:ManySuddenApprx)->Generator[TQobj, None, None]:
            for n in range(self.num_pts):
                yield self(torch.arange(n*self.niter_eval, (n+1)*self.niter_eval)*self.dt)
        return yield_time_evolve(self)
    
    def __call__(self, t:torch.Tensor, cum_:bool = True) -> TQobj:
        H = self.H(t)
        lam, U = H.eig(eigenvectors=True)
        lam:TQobj = TQobj(torch.diag_embed(lam), meta=U._metadata)
        if(cum_):
            return (U @ (lam*(t[1]-t[0])).expm() @ U.dag()).cummatprod()
        else:
            return (U @ (lam*(t[1]-t[0])).expm() @ U.dag()).matprodcontract()
    
    def evolve(self, a:float = 0.0, b:float = 1.0, num_pts:int|None = None)->TQobj:
        if(num_pts is None):
            num_pts = self.num_pts
        return self(torch.linspace(a,b,num_pts))