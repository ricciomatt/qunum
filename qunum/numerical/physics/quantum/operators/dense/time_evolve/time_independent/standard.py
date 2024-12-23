from .....qobjs.dense import TQobj, zeros_like, fofMatrix
from torch import Tensor, diag_embed
from .....qobjs.dense.core.qobj_einsum import einsum as qein

class TimeEvolve:
    def __init__(self, H:TQobj, hbar:float = 1.0) -> TQobj:
        self.lam_ij, self.U = H.diagonalize(inplace= False,ret_unitary=True,)
        self.hbar = hbar
        return
    
    def __call__(self, t:Tensor) -> TQobj:
        return self.U.dag() @ qein('A, ij->Aij', -(1j/self.hbar)*(t), self.lam_ij).exp() @ self.U

@fofMatrix(save_eigen = True, recompute=False)
def tEvolve(H:TQobj, t:Tensor, *args:tuple , hbar:float = 1, **kwargs:dict) -> TQobj:
    return qein('A, ij->Aij', -(1j/hbar)*(t), H).exp()