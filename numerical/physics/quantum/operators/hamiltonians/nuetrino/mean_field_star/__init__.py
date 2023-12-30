
import torch
from typing import Callable
from .obj import MeanField
from .phyics_functs import *
from ......data.hamiltonian import LazyTimeHamiltonian
from .......lattice_operators.integrators import NewtonCoates
from .mean_field import init_psi
def set_up_ham(omega0:float= 1.0, 
              omega_mult:torch.Tensor|None = None, 
              Rv:float= 50e6,
              r_0:float = 50e6, 
              v:float  = c, 
              mu0_mult:None|float=1e4, 
              num_particles:int = 3,
             order:int = 4, 
             dt:float = 1e-3,
             device:int|str = 0,
             operator:Callable= hamiltonian_operator,
             integrator = NewtonCoates(1, torch.complex128))->tuple[Callable, Callable, Callable]:
    if(omega_mult is None):
        omega_mult = torch.tensor(torch.arange(1,num_particles+1).numpy(), dtype = torch.complex64)
    omega = omega_mult*omega0
    mu0 = mu0_mult*omega0
    v = c
    Mf = MeanField(num_particles, omega=omega, mu0=mu0, Rv=Rv, r_0=r_0, v=v, device=device, operator=operator)
    H = LazyTimeHamiltonian(Mf, dt=dt)
    M = Magnus(H, order=order, dt = dt, Int=integrator)
    return M, H, Mf
