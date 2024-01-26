
import torch
from typing import Callable
from .obj import MeanField
from .phyics_functs import *
from ......data.hamiltonian import LazyTimeHamiltonian
from .......numerics.integrators import NewtonCoates
from .....qobjs import TQobj, direct_prod
from ....nuetrino import pmns2
import numpy as np 
from .......seml.fitting_algos import MagnusGenerator

def init_psi(n:int, theta:float = np.pi/5, dtype = torch.complex128)->TQobj:
    b = TQobj(torch.tensor([[1.,0.]], dtype= dtype), n_particles=1, hilbert_space_dims=2)
    Uh = pmns2(theta=torch.tensor(theta), dtype=dtype)
    #TQobj(torch.tensor([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]], dtype=torch.complex128), n_particles=1, hilbert_space_dims=2)
    args = (Uh @ b.dag() for i in range(n))
    args2 =  (Uh  for i in range(n))
    #Try to add back the 
    return direct_prod(*args), direct_prod(*tuple(args2))

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
             integrator = NewtonCoates(1, torch.complex128),
             dtype= torch.complex128)->tuple[Callable, Callable, Callable]:
    if(omega_mult is None):
        omega_mult = torch.tensor(torch.arange(1,num_particles+1).numpy(), dtype = torch.complex64)
    omega = omega_mult*omega0
    mu0 = mu0_mult*omega0
    v = c
    Mf = MeanField(num_particles, omega=omega, mu0=mu0, Rv=Rv, r_0=r_0, v=v, device=device, operator=operator,  dtype = dtype)
    H = LazyTimeHamiltonian(Mf, dt=dt)
    M = MagnusGenerator(H, order=order, dt = dt, Int=integrator)
    return M, H, Mf
