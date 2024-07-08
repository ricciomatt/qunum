from typing import Any
import torch
from torch import einsum
from ......constants import c as c

@torch.jit.script
def mu(t:torch.Tensor, mu0:float, Rv:float, r_0:float)->torch.Tensor:
    return mu0*( 1 - torch.sqrt(1 - ((Rv/(r_0 + t))**(2)) ) )**(2)

@torch.jit.script
def hamiltonian_operator(H_0:torch.Tensor, H_1:torch.Tensor, t:torch.Tensor, r:float, Rv:float, v:float, mu0:float )->torch.Tensor:
    u = mu(t*v, mu0, Rv, r)
    a = torch.ones(u.shape[0], dtype = torch.complex64)
    return (einsum('n, ij->nij', a.to(H_0.device), H_0) + einsum('n,ij->nij',(u), H_1))

@torch.jit.script
def hamiltonian_operator_exp(H_0:torch.Tensor, H_1:torch.Tensor, t:torch.Tensor, r:float, Rv:float, v:float, )->torch.Tensor:
    u = (Rv/r)*torch.exp(-t*v)
    a = torch.ones(u.shape[0], dtype = H_0.dtype, device = H_0.device)
    return (
            einsum('n, ij->nij', a.to(H_0.device), H_0) 
                + 
            einsum('n,ij->nij',(u), H_1)
        )
 
@torch.jit.script
def get_J(n:int, sigma:torch.Tensor, dtype:torch.dtype = torch.complex128)->torch.Tensor:
    J = torch.empty((n, 3, int(sigma.shape[1]**n), int(sigma.shape[1]**n)), dtype = dtype)
    for i in range(3):
        for j in range(n):
            if(j == 0):
                temp = sigma[i+1]
            else:
                temp = sigma[0]
            for k in range(1,n):
                if(k == j):
                    temp = torch.kron(temp, sigma[i+1])
                else:
                    temp = torch.kron(temp, sigma[0])
            J[j,i] = temp
    return J/2


@torch.jit.script
def H0(omega:torch.Tensor, J:torch.Tensor)->torch.Tensor:
    return einsum('w, wij -> ij', -omega, J[:,2])
    
@torch.jit.script
def H1(J:torch.Tensor)->torch.Tensor:
    H_1 = torch.zeros((J.shape[2:]), dtype = J.dtype)
    for p in range(J.shape[0]):
        for q in range(J.shape[0]):
            if(p!=q):
                H_1 += einsum('mij, mjk-> ik', J[p], J[q])
    return H_1
