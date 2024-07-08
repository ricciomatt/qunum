from typing import Callable
import torch
from .phyics_functs import hamiltonian_operator, H0, H1, get_J
from torch import Tensor
from .......mathematics.algebra.representations.su import get_gellmann, get_pauli
from .....qobjs import TQobj
from IPython.display import display as disp, Markdown as md
from sympy import latex, Matrix, I
from ......constants import c
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
                operator:Callable= hamiltonian_operator,
                dtype:torch.dtype = torch.complex128
                ):
        
        self.N = float(n_particles)
        self.mu0 = float(mu0)
        self.v = float(v)
        self.r0= float(r_0)
        self.Rv = float(Rv)
        J = self.getJ(dtype)
        if(omega is not None):
            self.w = omega
            try:
                self.w0 = omega.real.min()
            except:
                self.w0 = omega[0]
        else:
            self.w = torch.arange(1, n_particles+1)*omega_0
            self.w = torch.complex(self.w, torch.zeros_like(self.w))
            self.w0 = omega_0
            
        self.w = self.w.type(J.dtype)
        print(J.dtype, self.w.dtype)
        self.H0 = H0(self.w, J)
        self.H1 = H1(J)
        
        self.flavors = flavors
        self.device = device
        self.H1/=self.N 
        self.O = operator
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
        return self.O(self.H0, self.H1, t, self.r0, self.Rv, self.v, self.mu0)
    
    def to(self, device:str|int):
        self.device = device
        self.w.to(device)
        self.H0.to(device)
        self.H1.to(device)
        return 
    
    def getJ(self, dtype)->Tensor:
        J = get_J(int(self.N), get_pauli(to_tensor= True), dtype=dtype)
        return J
    
    def getHBasis(self)->TQobj:
        return {TQobj(self.H0/self.H0.norm(), hilbert_space_dims=self.flavors, n_particles=int(self.N)), 
                TQobj(self.H1/self.H1.norm(), hilbert_space_dims=self.flavors, n_particles=int(self.N))}
         
    
    def __repr__(self):
        disp(md(f'''$$\\omega_i = {latex(Matrix(self.w.detach().real.numpy()).T)}\\\\v={str(self.v)}, R_\\nu={str('{:.2e}'.format(self.Rv))}, r_0 = {str('{:.2e}'.format(self.r0))}, \\mu_0 = {str(self.mu0)}$$'''))
        disp(md(f'''$$\\mathcal{'{H}'}_{'{0}'} = {latex(Matrix(self.H0.real.detach().numpy())+ I *Matrix(self.H0.imag.detach().numpy()))}$$'''))
        disp(md(f'''$$\\mathcal{'{H}'}_{'{1}'} = {latex(Matrix(self.H1.real.detach().numpy())+ I *Matrix(self.H1.imag.detach().numpy()))}$$'''))
        return f'{str(int(self.N))} Particle {str(int(self.flavors))} Flavor, Mean Field Neutrino Hamiltonian. '
    
    def get_mu(self, t:Tensor)->Tensor:
        return  self.mu0*torch.pow(1-torch.sqrt(1-torch.pow(self.Rv/(self.r0+self.v*t),2)),2)
