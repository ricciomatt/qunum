from torch import Tensor
from ......quantum.qobjs import TQobj
from ......data.hamiltonian import LazyTimeHamiltonian
from .......mathematics import D_Op
class TimeSchrodingerEq:
    def __init__(self, **kwargs):
        self.D = D_Op(**kwargs)
        pass
    def __call__(self, U:TQobj, t:Tensor, H:TQobj):
        return self.D(U, t) - H @ U
    