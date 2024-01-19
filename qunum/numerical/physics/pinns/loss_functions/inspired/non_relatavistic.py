import torch
from torch import einsum
from torch.autograd import grad as AutoGrad
from typing import Callable,  List, Optional
from ....data.hamiltonian import LazyTimeHamiltonian
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

class ShrodingerEqLossLoss:
    def __init__(self,
                 H:LazyTimeHamiltonian,
                 time_dim:int = 0)->None:
        self.H = H
        self.time_dim = time_dim
        return
    def __call__(self, Uh:torch.Tensor, U:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        l = DtU(Uh, t, time_dim=self.time_dim) - 1j*self.H(t) @ Uh
        return (l.dag() @ l).sum()
                     
@nb.jit(forceobj=True)
def dt(A: torch.Tensor, Inpt: torch.Tensor) -> torch.Tensor:
    dU_dt = torch.empty_like(
        A)
    
    for i in range(A.shape[1]):
        
        dU_dt[:,i] = AutoGrad(
            A[:, i],  # Derivative with respect to the i-th column of A
            Inpt,
            grad_outputs=torch.ones_like(A[:,i]),
            allow_unused=True, 
            create_graph=True, 
            retain_graph=True
        )[0][:,-1]
    return dU_dt

from typing import List, Optional

@torch.jit.script
def DtU(U: torch.Tensor, t: torch.Tensor, time_dim: int) -> torch.Tensor:
    # Create an empty tensor with the same shape as U to store the result
    DU = torch.empty_like(U)
    # Loop over dimensions of U
    for i in range(U.shape[1]):
        for j in range(U.shape[2]):
            # Initialize grad_outputs as a list containing a tensor of ones
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(t)]
            
            # Compute the gradient using AutoGrad
            temp: Optional[torch.Tensor] = AutoGrad(
                [U[:, i, j]],
                [t],
                grad_outputs=grad_outputs,
                retain_graph=True,
                allow_unused=True,
                create_graph=True,
            )[0]
            # Check if the gradient is not None
            if temp is not None:
                # If not None, store the conjugate of the gradient in DU
                DU[:, i, j] = temp[:, time_dim].conj()
            else:
                # If None, fill DU with zeros
                DU[:, i, j] = torch.zeros_like(t)
    
    return DU