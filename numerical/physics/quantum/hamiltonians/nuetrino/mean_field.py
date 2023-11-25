from typing import Any
import torch
from sympy import Matrix, latex, I
from torch import einsum
import numba as nb
from qutip import tensor, basis, Qobj
import numpy as np
from .....constants import c as c
from .....seml.fitting_algos import Magnus
from .....algebra.representations.su import get_pauli
from .....seml.data.data_loaders.physics.quantum import LazyTimeHamiltonian
from IPython.display import display as disp, Markdown as md, Math as mt

class MeanField:
    def __init__(self, 
                n_particles:int=2,
                flavors:int = 2,
                omega:torch.Tensor|None = None,
                omega_0:float|None = 1.,
                mu0:float|torch.Tensor = 5.,
                Rv:float|torch.Tensor = 50e6,
                r_0:float = 50e6,
                v:float = c,
                device:int|str = '0',
                ):
        
        J = get_J(n_particles, get_pauli(n_particles))
        if(omega is not None):
            self.w = omega
            self.w0 = omega.min()
        else:
            self.w = torch.arange(1, n_particles+1)*omega_0
            self.w = torch.complex(self.w, torch.zeros_like(self.w))
            self.w0 = omega_0
        
        self.w.type(torch.complex64)
        self.H0 = H0(self.w, J)
        self.H1 = H1(J)
        self.mu0 = float(mu0)
        self.v = float(v)
        self.r = float(r_0)
        self.Rv = float(Rv)
        self.N = float(n_particles)
        self.flavors = flavors
        self.device = device
        
        try:
            self.H0.to(device)
            self.H1.to(device)
            self.N.to(device)
            self.v.to(device)
            self.r.to(device)
            self.Rv.to(device)
        except:
            pass
        return  
    
    def __call__(self, t:torch.Tensor) -> torch.Tensor:
        return hamiltonian_operator(self.H0, self.H1, t, self.r, self.Rv, self.N, self.mu0)
    
    def to(self, device:str|int):
        self.device = device
        self.w.to(device)
        self.H0.to(device)
        self.H1.to(device)
        return 
        
    
    def __repr__(self):
        disp(md(f'''$$\\omega_i = {latex(Matrix(self.w.detach().real.numpy()).T)}\\\\v={str(self.v)}, R_\\nu={str('{:.2e}'.format(self.Rv))}, r_0 = {str('{:.2e}'.format(self.Rv))}, \\mu_0 = {str(self.mu0)}$$'''))
        disp(md(f'''$$\\mathcal{'{H}'}_{'{0}'} = {latex(Matrix(self.H0.real.detach().numpy())+ I *Matrix(self.H0.imag.detach().numpy()))}$$'''))
        disp(md(f'''$$\\mathcal{'{H}'}_{'{1}'} = {latex(Matrix(self.H1.real.detach().numpy())+ I *Matrix(self.H1.imag.detach().numpy()))}$$'''))
        return f'{str(int(self.N))} Particle {str(int(self.flavors))} Flavor, Mean Field Neutrino Hamiltonian. '
        
@torch.jit.script
def hamiltonian_operator(H_0:torch.Tensor, H_1:torch.Tensor, t:torch.Tensor, r:float, Rv:float, v:float, N:float, mu0:float)->torch.Tensor:
    u = mu0/N*torch.pow(1-torch.sqrt(1-torch.pow(Rv/(r+v*t),2)),2)
    a = torch.ones(u.shape[0], dtype = torch.complex64)
    return (einsum('n, ij->nij', a.to(H_0.device), H_0) + einsum('n,ij->nij',(u), H_1))


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
                         hbar:float = 1, 
                         device:int = 0)->torch.Tensor:
    u = mu(t*c, mu0, Rv, r_0)
    a = torch.ones(u.shape[0], dtype = torch.complex64)
    return (einsum('n, ij->nij', a.to(device), H_0.clone()) + einsum('n,ij->nij',(u/m).to(device), H_1.clone()))


def set_up_ham(omega0:float= 1.0, 
              omega_mult:torch.Tensor|None = None, 
              Rv:float= 50e6,
              r_0:float = 50e6, 
              v:float  = c, 
              mu0_mult:None|float=10., 
              num_particles:int = 3,
             order:int = 4, 
             dt:float = 1e-3,
             device:int|str = 0)->torch.Tensor:
    if(omega_mult is None):
        omega_mult = torch.tensor(torch.arange(1,num_particles+1).numpy(), dtype = torch.complex64)
    omega = omega_mult*omega0
    mu0 = mu0_mult*omega0
    v = c
    Mf = MeanField(num_particles, omega=omega, mu0=mu0, Rv=Rv, r_0=r_0, v=v, device=device)
    H = LazyTimeHamiltonian(Mf, dt=dt)
    M = Magnus(H, order=order, dt = dt)
    return M, H