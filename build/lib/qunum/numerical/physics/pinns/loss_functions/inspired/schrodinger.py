from ....data import LazyTimeHamiltonian
from torch import Tensor
from ....quantum import TQobj
from .....mathematics import Dx
class SchrodingerEqLossLoss:
    def __init__(self,
                 H:LazyTimeHamiltonian,
                 time_dim:int = 0)->None:
        self.H = H
        self.time_dim = time_dim
        return
    def __call__(self,*args, **kwargs)->Tensor:
        Uh = args[0]
        t = args[2]
        l = Dx(Uh, t, time_dim=self.time_dim, retain_graph=True, create_graph=False, allow_unused=True, symmetric=False) - 1j*self.H(t) @ Uh
        return (l.dag() @ l).sum()
                  