import torch 
from torch import einsum
from ..interpolation.polynomial import lagrange_interp_coef

class NewtonCoates:
    def __init__(self, order:int, dtype = None)->None:
        self.order = order
        if(order !=1):
            self.L = lagrange_interp_coef(torch.linspace(0, 1, self.order))
        else:
            self.L = torch.tensor([1.])
        if(dtype is not None):
            self.L.type(dtype)
        return
    def __call__(self, f:torch.Tensor)->torch.Tensor:
        return integrate_newton_coates_do(f, self.L)
    def cumeval(self, f:torch.Tensor, dx:float)->torch.Tensor:
        M = self.__call__(f,)
        return M.cumsum(dim=0) * dx 
    def eval(self, f, dx)->torch.Tensor:
        M = self.__call__(f,)
        return M.sum(dim=0) * dx 

@torch.jit.script
def integrate_newton_coates_do(f:torch.Tensor, L:torch.Tensor)->torch.Tensor:
    M = torch.zeros_like(f)
    for i in range(L.shape[0]):
        M[L.shape[0]:]+=f[i: f.shape[0] - L.shape[0] + i] * L[i]
    M[:L.shape[0]] = f[:L.shape[0]]
    return M


