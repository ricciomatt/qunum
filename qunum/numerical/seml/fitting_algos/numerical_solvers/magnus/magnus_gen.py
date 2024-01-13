from typing import Any
import torch
from .....physics.data.hamiltonian import LazyTimeHamiltonian
import numpy as np
from .....finite_element.integrators import NewtonCoates, integrate_newton_coates_do as newton
from scipy.special import bernoulli
from torch import einsum
from .....algebra import ad
from .....physics.quantum import TQobj
from IPython.display import display as disp, Math as Mt

class MagnusGenerator:
    def __init__(self, 
                 Hamiltonian:LazyTimeHamiltonian,
                 order:int=2, 
                 ix0:int = 0, 
                 dt:float = 1e-3,
                 num_int:int = int(5e1),
                 Int:NewtonCoates= NewtonCoates(2, dtype=torch.complex128),
                 set_iter_len:int=int(1e3),
                 call_funct = 'gen_function'
                 )->None:
        self.ix0 = ix0
        self.H = Hamiltonian
        self.set_order(order)
        self.Int = Int
        self.n = 0
        self.num_int = num_int
        self.dt = dt
        self.iter = 1
        self.iter_len = set_iter_len
        self.call_funct = call_funct
        return
    
    def reset_iter(self):
        self.n = 0
        self.iter = 1
        return 
    
    def set_order(self, order:int)->None:
        self.order = order
        self.Bk = torch.tensor(
            bernoulli(order),
            dtype=torch.complex128
        )
        return
    
    def gen_function(self, 
                 a:float=0., 
                 b:float=1.,
                 num_pts:int = int(1e2),
                 U0:None|torch.Tensor = None,
                 raw_omega:bool = False)->torch.Tensor:
        dx = (b-a)/num_pts
        x = torch.linspace(a, b, num_pts)
        H = self.H(x)
        Omega = expansion(H, self.Bk, self.order, dx, self.Int.L)
        if(raw_omega):
            del H
            return Omega
            
        elif(U0 is None):
            U0 = torch.eye(H.shape[1], H.shape[1], dtype= H.dtype)
            del H
            Omega = Omega.sum(dim = 0)
            Omega = TQobj(Omega)
            return Omega.expm().cummatprod() @ U0
    
    def __call__(self, 
                 a:float=0., 
                 b:float=1.,
                 num_pts:int = int(1e2),
                 U0:None|torch.Tensor = None,
                 raw_omega:bool = False,
                )->torch.Tensor:
        return getattr(self, self.call_funct)(a, b, num_pts, U0, raw_omega=raw_omega)
    
    def __next__(self)->torch.Tensor:
        if(self.n<self.iter*self.iter_len):
            H = self.H(torch.linspace(self.n*self.dt - self.dt/2, (self.n+1/2)*self.dt, self.num_int))
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
    
    
@torch.jit.script
def comm(A:torch.Tensor,B:torch.Tensor):
    return torch.einsum('Aij, Ajk-> Aik', A, B) - torch.einsum('Aij, Ajk-> Aik', B, A)

@torch.jit.script
def tpcomm(A:torch.Tensor,B:torch.Tensor):
    return torch.einsum('Aij, Bjk-> ABik', A, B) - torch.einsum('Aij, Bjk-> ABik', B, A)


@torch.jit.script
def expansion(H:torch.Tensor, 
              Bk:torch.Tensor, 
              order:int,
              dx:float,
              L:torch.Tensor,)->torch.Tensor:
    H *= -1j
    Omega = torch.zeros((order, H.shape[0], H.shape[1], H.shape[1]), dtype = H.dtype)
    Omega[0] = newton(H.clone(), L).cumsum(dim=0)*dx
    S = torch.zeros((order, order, H.shape[0], H.shape[1], H.shape[1]), dtype = H.dtype)
    if(order >= 2):
        for k in range(2, order+1):
            n = k-1
            for i in range(1, k):
                j = i-1
                if(i == 1):
                    t = H.clone()
                    S[n, j] = comm(Omega[n-1], t)
                elif(i == n):
                    S[n, j] = ad(Omega[0], H.clone(), j)
                    
                else:
                    for m in range(1, n-j):
                        S[n, j] += comm(Omega[n-m], S[n-m,j-1])
                Omega[n] +=  Bk[i]/torch.math.factorial(i)*newton(S[n,j], L).cumsum(dim = 0)*dx
    return Omega

#@torch.jit.script
def expansion_show(H:torch.Tensor, 
              Bk:torch.Tensor, 
              order:int,
              dx:float,
              L:torch.Tensor, 
              disp_:bool = True)->torch.Tensor:
    H *= -1j
    
    print(newton(H.clone(), L).cumsum(dim=0)*dx)
    print(L, H.sum(dim=0)*dx)
    Omega = torch.zeros((order, H.shape[0], H.shape[1], H.shape[1]), dtype = torch.complex64)
    Omega[0] = newton(H.clone(), L).cumsum(dim=0)*dx
    S = torch.zeros((order, order, H.shape[0], H.shape[1], H.shape[1]), dtype = torch.complex64)
    if(order >= 2):
        for k in range(2, order+1):
            n = k-1
            st = f'\\Omega_{k} = '
            for i in range(1, k):
                j = i-1
                if(i == 1):
                    st+='+ \\frac{B_{'+str(i)+'}}{{'+str(i)+'}!}'
                    t = H.clone()
                    S[n,j] = comm(Omega[n-1].clone(), t)
                    print(S[n,j][-1])
                    print(Omega[n-1][-1])
                    print()
                    print()
                    st+='[\\Omega_{'+str(n)+'}, (-iH)]'
                elif(i == n):
                    st+='+ \\frac{B_{'+str(i)+'}}{{'+str(i)+'}!}'
                    S[n,j] = ad(Omega[0].clone(), H.clone(), j)
                    
                    print(S[n,j,-1])
                    st+='ad^{'+str(j+1)+'}_{\\Omega_{1}} (-iH) + '
                else:
                    st+='+ \\frac{B_{'+str(i)+'}}{{'+str(i)+'}!}'
                    for m in range(1, n-j):
                        S[n, j] += comm(Omega[n-m].clone(), S[n-m,j-1].clone())
                        st+='[\\Omega_{'+str(n-m)+'}, S_{'+str(n-m)+'}^{'+str(j)+'}]+'
                    st = st.rstrip('+')
                Omega[n] +=  Bk[i]/torch.math.factorial(i)*newton(S[n,j], L).cumsum(dim = 0)*dx
                
            if(disp_):
                disp(Mt(st))
                disp(Bk)
                print('\n\n\n\n\n\n')
                
    return Omega



    
