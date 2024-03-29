from torch import Tensor
from ......physics.quantum.qobjs import TQobj
from .....data.data_loaders.lazy.hamiltonian import LazyTimeHamiltonian
from ......mathematics.autograd.grad_obj import D_Op
class TimeSchrodingerEq:
    def __init__(self, **kwargs):
        self.D = D_Op(**kwargs)
        pass
    def __call__(self, U:TQobj, t:Tensor, H:TQobj):
        return self.D(U, t) - H @ U
    