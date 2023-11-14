from typing import Any
import torch
from ....data.data_loaders.physics.quantum.hamiltonian import LazyTimeHamiltonian
import numpy as np
from .....lattice_operators.integrators import NewtonCoates, integrate_newton_coates_do as newton
from scipy.special import bernoulli
from torch import einsum

class Magnus:
    def __init__(self, 
                 Hamiltonian:LazyTimeHamiltonian,
                 order:int=2, 
                 ix0:int = 0, 
                 dt:float = 1e-3,
                 num_int:int = int(5e1),
                 Int:NewtonCoates= NewtonCoates(2, dtype=torch.complex64),
                 set_iter_len:int=int(1e3))->None:
        self.ix0 = ix0
        self.H = Hamiltonian
        self.set_order(order)
        self.Int = Int
        self.n = 0
        self.num_int = num_int
        self.dt = dt
        self.iter = 1
        self.iter_len = set_iter_len
        return
    
    def reset_iter(self):
        self.n = 0
        self.iter = 1
        return 
    
    def set_order(self, order:int)->None:
        self.order = order
        self.Bk = torch.tensor(
            bernoulli(order),
            dtype=torch.complex64
        )
        return
    
    def __call__(self, 
                 a:float=0., 
                 b:float=1.,
                 num_pts:int = int(1e2),
                 U0:None|torch.Tensor = None,
                 raw_omega:bool = False,
                )->torch.Tensor:
        dx = (b-a)/num_pts
        x = torch.linspace(a, b, num_pts)
        H = self.H(x)
        Omega = expansion(H, self.Bk, self.order, dx, self.Int.L)
        if(raw_omega):
            del H
            return Omega
        elif(U0 is None):
            I = torch.complex(0,1)**torch.arange(1, self.order+1)
            U0 = torch.eye(H.shape[1], H.shape[1], dtype= torch.complex64)
            del H
        return torch.linalg.matrix_exp(torch.einsum('o, Aojk->Ajk', I, Omega[1:])) @ U0
    
    def __next__(self)->torch.Tensor:
        if(self.n<self.iter*self.iter_len):
            H = self.H(torch.linspace(self.n*self.dt, (self.n+1)*self.dt, self.num_int))
            Omega = expansion(H, self.Bk, self.order, self.dt/self.num_int, self.Int.L)
            self.n+=1
            del H 
            return Omega[:,Omega.shape[1]-1]
        else:
            raise StopIteration
    
    def __getitem__(self, ix:int) ->torch.Tensor:
        H = self.H(torch.linspace(ix*self.dt, (ix)*self.dt, self.num_int))
        Omega = expansion(H, self.Bk, self.order, self.dt/self.num_int, self.Int.L)
        del H
        self.n+=1
        return Omega[:,Omega.shape[1]-1]
    
    def __iter__(self)->object:
        return self
    
    def reset_iter(self)->None:
        self.n = 0
        return 
    
    def set_iterations(self,n:int|None = None, iter_len:int|None = None, iteration_:None|int = None)->None:
        if(n is not None):
            self.n = n
        if(iter_len is not None):
            self.iter_len = iter_len
        if(iteration_ is not None):
            self.iter= iteration_
        return
    
    def fourth_order(self,
                    a:float=0.,
                    b:float=1.,
                    num_pts:int=1e2,
                    U0:None|torch.Tensor=None,
                    raw_omega:bool = False
                    )->torch.Tensor:
        dx = (b-a)/num_pts
        H = -1j*self.H(torch.linspace(a,b,num_pts, dtype=torch.complex64))
        Omega = torch.empty((4, H.shape[0], H.shape[1], H.shape[2]), dtype=torch.complex64)    
        Omega[0] = self.Int.cumeval(H, dx)
        Omega[1] = (1/2)*self.Int.cumeval(comm(H, Omega[0]), dx)
        Omega[2] = (1/6)*(self.Int.cumeval(comm(H, Omega[1]),dx) + self.Int.cumeval(torch.einsum('Aij, Ajk, Akm-> Aim', Omega[0], H, H), dx)) 
        Omega[3] = (1/12)*()
        return Omega
    
@torch.jit.script
def comm(A:torch.Tensor,B:torch.Tensor):
    return torch.einsum('Aij, Ajk-> Aik', A, B) - torch.einsum('Aij, Ajk-> Aik', B, A)

@torch.jit.script
def tpcomm(A:torch.Tensor,B:torch.Tensor):
    return torch.einsum('Aij, Bjk-> ABik', A, B) - torch.einsum('Aij, Bjk-> ABik', B, A)

    

# have to fix this function. think this works have to try on nuetrino hamiltonian
@torch.jit.script
def expansion(H:torch.Tensor, 
              Bk:torch.Tensor, 
              order:int,
              dx:float,
              L:torch.Tensor)->torch.Tensor:
    H *= -1j
    Omega = torch.zeros((order, H.shape[0], H.shape[1], H.shape[1]), dtype = torch.complex64)
    Omega[0] = newton(H.clone(), L).cumsum(dim=0)*dx
    Omega[0] = H.clone().cumsum(dim=0)*dx
    S = torch.zeros((order, order, H.shape[0], H.shape[1], H.shape[1]), dtype = torch.complex64)
    if(order >= 2):
        for k in range(2, order+1):
            n = k-1
            for i in range(1, k):
                j = i-1
                if(i == 1):
                    t = H.clone()
                    S[n,j] += comm(Omega[n-1], t)
                
                elif(i == n):
                    t = H.clone()
                    for m in range(k-1):
                        S[n,j] = comm(Omega[0], t)
                        t = S[n,j].clone()
                else:
                    for m in range(1, i-k):
                        print(m)
                        S[n, j] += comm(Omega[n-m], S[n-m,j-1])
                
                Omega[n] +=  Bk[i]/torch.math.factorial(i) * S[n,j].cumsum(dim = 0)*dx
               
    return Omega

        


    
    
