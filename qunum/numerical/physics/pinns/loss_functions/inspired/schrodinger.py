from ....data import LazyTimeHamiltonian
from torch import Tensor
from ....quantum import TQobj
from .....mathematics import Dx
class SchrodingerEqLossLoss:
    def __init__(self,
                 H:LazyTimeHamiltonian,
                 time_dim = 0)->None:
        self.H = H
        self.time_dim = time_dim
        return
    def __call__(self, Uh, t, *args, **kwargs)->Tensor:
        l = Dx(Uh, t, der_dim=self.time_dim, retain_graph=True, create_graph=False, allow_unused=True, symmetric=False) - 1j*self.H(t[:,self.time_dim]) @ Uh
        return (l.dag() @ l).sum()
                  