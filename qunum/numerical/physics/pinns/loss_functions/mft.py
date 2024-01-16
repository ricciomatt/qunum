from .std import ComplexMSE
from .inspired import HamiltonianLossFunction, LazyTimeHamiltonian
from typing import Callable
import torch
def default_weighting(n:int, alpha:float, lam_0:torch.Tensor):
    A = torch.exp(-n*alpha)
    lam_0[0]*=A/2
    lam_0[1]*=(1-A)
    lam_0[2]*=A/2
    pass

class PinnMagnusLf:
    def __init__(self, 
                 H:LazyTimeHamiltonian, 
                 get_weigthing:Callable[[int, float], torch.Tensor]|None= None, 
                 alpha:float = 1e-3, 
                 n_particles:int =  2):
        self.InspiredLoss = HamiltonianLossFunction(H, n_particles)
        self.InitialConditionsLoss = lambda y, yh, x, *args: (y-torch.eye(y.shape[-2], y.shape[-1], dtype = y.dtype))
        self.SimulationLoss = ComplexMSE()
        if(get_weigthing is None):
            self.get_weigthing = lambda n, alpha: default_weighting(n, alpha, torch.ones(3))
        else:
            self.get_weigthing = get_weigthing
        self.n = 0
        self.alpha = alpha
        return
    def __call__(self, yh, y, x, *args, iter_:bool= True, **kwargs)->torch.Tensor:
        self.InspiredLoss(yh, y, *args)
        lambd = self.get_weigthing(self.n, self.alpha)
        if(iter_):
            self.n+=1
        return lambd[0] * self.SimulationLoss(yh, y, x, *args) + lambd[1] * self.InspiredLoss(yh, y, x, *args) + lambd[2] * self.InitialConditionsLoss(yh, y, x, *args)