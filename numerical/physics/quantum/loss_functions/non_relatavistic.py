import torch
from torch import autograd as AutoGrad, einsum
from typing import Callable
from ....seml.data.data_loaders import LazyTimeHamiltonian
import numba as nb

class TimeIndependentHamiltonain:
    def __init__(self, V_x:Callable, En:float):
        self.V = V_x
        self.En = En
        return
    def __call__(self, psi:torch.Tensor, x:torch.Tensor)->torch.Tensor:
        psi.backward(torch.ones_like(psi),retain_graph=True)
        px = AutoGrad.grad(psi, x, grad_outputs=torch.ones_like(x), allow_unused=True, create_graph=True, retain_graph=True)[0]
        return self.En - (1/2)*px @ px.T - self.V(x)

class TimeDependentHamiltonain:
    def __init__(self, V_x:Callable,):
        self.V = V_x
        return
    
    def __call__(self, psi:torch.Tensor, x:torch.Tensor)->torch.Tensor:
        psi.backward(torch.ones_like(psi),retain_graph=True)
        px = AutoGrad.grad(psi, x[1:], grad_outputs=torch.ones_like(x[1:]), allow_unused=True, create_graph=True, retain_graph=True)[0]
        p0 = AutoGrad.grad(psi, x[0], grad_outputs=torch.ones_like(x[0]), allow_unused=True, create_graph=True, retain_graph=True)[0]
        return torch.complex(0,1) * p0 + (1/2)*px @ px.T + self.V(x)
    
    
class HamiltonianLossFunction:
    def __init__(self, 
                 H:LazyTimeHamiltonian, 
                 ndims:int)->None:
        self.H = H
        self.n = ndims 
        return 
    def __call__(self, yh:torch.Tensor, y:torch.Tensor, Inpt:torch.Tensor)->torch.Tensor:
        l = einsum('Aij, Aj -> Ai',
                   -1j*self.H(Inpt[:, -1]).to(Inpt.device), 
                   yh) - dt(
                       yh, 
                       Inpt
                    )
        return einsum('Ai, Ai -> ', l, l.conj())
    
    
@nb.jit(forceobj=True)
def dt(A: torch.Tensor, Inpt: torch.Tensor) -> torch.Tensor:
    dU_dt = torch.empty_like(A)
    for i in range(A.shape[1]):
        dU_dt[:,i] = AutoGrad.grad(
            A[:, i],  # Derivative with respect to the i-th column of A
            Inpt,
            grad_outputs=torch.ones_like(A[:,i]),
            allow_unused=True, 
            create_graph=True, 
            retain_graph=True
        )[0][:,-1]
    return dU_dt