import torch
from torch import einsum
import numba as nb
from plotly import express as px, graph_objects as go, subplots as plt_sub
from plotly.offline import iplot, init_notebook_mode
from qutip import tensor, basis, Qobj
import numpy as np
init_notebook_mode(True)
import sys,os
sys.path.append(f'/home/{os.getlogin()}')
from ....const_and_mat.constants import c as c
from ....seml.fitting_algos import Magnus
from ....algebra.representations.su import get_pauli
from ....seml.data.data_loaders.physics.quantum import LazyTimeHamiltonian


@torch.jit.script
def get_J(n:int, sigma:torch.Tensor)->torch.Tensor:
    J = torch.empty((n, 3, int(sigma.shape[1]**n), int(sigma.shape[1]**n)), dtype = torch.complex64)
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

@nb.jit(forceobj = True)
def init_psi(n:int, theta:float = np.pi/5)->Qobj:
    b = basis(2,0)
    Uh = Qobj(np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]))
    return tensor([Uh*b for i in range(n)])

@torch.jit.script
def H0(omega:torch.Tensor, J:torch.Tensor)->torch.Tensor:
    return einsum('w, wij -> ij', -omega, J[:,2])
    
@torch.jit.script
def H1(J:torch.Tensor)->torch.Tensor:
    H_1 = torch.zeros((J.shape[2:]), dtype = J.dtype)
    for p in range(J.shape[0]):
        for q in range(J.shape[0]):
            if(p<q):
                H_1 += einsum('mij, mjk-> ik', J[p], J[q])
    return H_1


@torch.jit.script
def mu(t:torch.Tensor, mu0:float, Rv:float, r_0:float)->torch.Tensor:
    return mu0*( 1 - torch.sqrt(1 - ((Rv/(r_0 + t))**(2)) ) )**(2)

@torch.jit.script
def nuetrino_hamiltonian(t:torch.Tensor,
                         H_0:torch.Tensor,
                         H_1:torch.Tensor,
                         c:float, 
                         m:int,
                         Rv:float,
                         r_0:float, 
                         mu0:float,
                         hbar:float = 1, )->torch.Tensor:
    u = mu(t*c, mu0, Rv, r_0)
    a = torch.ones(u.shape[0], dtype = torch.complex64)
    return (einsum('n, ij->nij', a, H_0.clone()) + einsum('n,ij->nij',u/m, H_1.clone()))


def set_up_ham(omega0:float= 1.0, 
              omega_mult:torch.Tensor|None = None, 
              Rv:float= 50e6,
              r_0:float = 50e6, 
              v:float  = c, 
              mu0_mult:None|float=10., 
              num_particles:int = 3,
             order:int = 4, 
             dt:float = 1e-3,)->torch.Tensor:
    if(omega_mult is None):
        omega_mult = torch.tensor(torch.arange(1,num_particles+1).numpy(), dtype = torch.complex64)
    omega = omega_mult*omega0
    sigma = get_pauli(to_tensor=True)
    mu0 = mu0_mult*omega0
    
    print('\\mu_{0}='+str(mu0)+',')
    
    v = c
    J = get_J(num_particles, sigma)
    H_0 = H0(omega, J)
    H_1 = H1(J)
    h = lambda t: nuetrino_hamiltonian(t, H_0, H_1, v, num_particles, Rv, r_0, mu0)
    
    H = LazyTimeHamiltonian(h)
    M = Magnus(H, order=order, dt = dt)
    return M, H