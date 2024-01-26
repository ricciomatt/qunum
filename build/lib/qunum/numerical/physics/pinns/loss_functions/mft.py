from .std import ComplexMSE
from .inspired import SchrodingerEqLossLoss
from ...data import LazyTimeHamiltonian
from typing import Callable, Generator
from torch import Tensor, ones, from_numpy
from numpy import array
def default_weighting(alpha:float, lam_0:Tensor):
    from math import exp
    n = 0
    while True:
        A = exp(-n*alpha)
        n+=1
        yield lam_0 * from_numpy(array([A, 1-A]))
class PinnMagnusLf:
    def __init__(
            self, 
            H:LazyTimeHamiltonian, 
            get_weigthing:Generator[Tensor, None, None]|None= None, 
            alpha:float = 1e-3, 
            n_particles:int =  2
        )->None:
        self.InspiredLoss = SchrodingerEqLossLoss(H, n_particles)
        self.SimulationLoss = ComplexMSE()
        if(get_weigthing is None):
            self.get_weigthing = default_weighting(alpha, ones(2))
        else:
            self.get_weigthing = get_weigthing
        self.n = 0
        self.alpha = alpha
        self.weight = next(self.get_weigthing)
        return
    
    def batch_update(self)->None:
        self.weight = next(self.get_weigthing)
        return 
    
    def reset_(self)->None:
        self.get_weigthing = default_weighting(self.alpha, ones(2))
        return
    
    def forward(self,
                Model:Callable,
                y:Tensor,
                x:Tensor,
        )->Tensor:
        return self.__call__(Model, y, x)
    
    def __call__(
            self, 
            Model:Callable,
            y:Tensor, 
            x:Tensor,
        )->Tensor:
        yh = tuple(map(lambda i: Model.forward(i), x))
        IL = sum(map(lambda a: self.InspiredLoss(a[0], a[1]), zip(x,yh)))
        
        return self.weight[0] * self.SimulationLoss(yh, y, None) + self.weight[1] * IL