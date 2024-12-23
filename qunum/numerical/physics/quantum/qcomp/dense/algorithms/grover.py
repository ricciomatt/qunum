import numpy as np 
import numpy.typing as npt
import torch
from ......mathematics.algebra.representations.su import get_pauli



sigma = get_pauli(to_tensor = True)
def CkiRz(theta:npt.NDArray, k):
    n = 2**(k + 1) 
    print(n)
    temp = np.ones(n, dtype=np.complex64)
    temp[-2] = 1j*np.exp(-1j * theta / 2)
    temp[-1] = 1j*np.exp(1j * theta / 2)
    return np.diag(temp)

@torch.jit.script
def CkiRz_torch(theta:torch.Tensor, k:int):
    n = int(2**(k+1))
    temp = torch.ones(n, theta.shape[0], 
                      dtype = torch.complex64)
    temp[-2:]= -1j*torch.exp(torch.einsum('A, j-> jA', theta/2, torch.complex(torch.zeros(2), torch.tensor([1.,-1.]))))
    return torch.einsum('ij, jA->Aij', torch.eye(n,n, dtype = torch.complex64), temp)


@torch.jit.script
def Hadamard()->torch.Tensor:
    H = torch.ones(2,2, dtype = torch.complex64)
    H[1,1] = -1
    return H/torch.sqrt(torch.tensor(2))


@torch.jit.script
def KronProdSum(O:torch.Tensor,n:int)->torch.Tensor:
    Hv = O.clone()
    for i in range(1, int(n)):
        Hv = torch.kron(Hv, O)
    return Hv

@torch.jit.script
def Uperp(n:int, 
        CkZ:torch.Tensor,
         sig:torch.Tensor)->torch.Tensor:
    H = Hadamard()
    H = KronProdSum(H, n)
    Xv = KronProdSum(sig, n)
    return torch.einsum('ij, jm, Amn, nk, kl->Ail', H, Xv, CkZ.detach().clone(), Xv,  H)

@torch.jit.script
def Uf(target_state:int,
       n:int, 
       CkZ:torch.Tensor,
       X:torch.Tensor):
    t = KronProdSum(X, n)
    return torch.einsum('ij, Ajk, kl->Ail', t, CkZ, t)


class GroverIteratorCkiRzEr:
    def __init__(self, 
                 n:int=2, 
                 target_state:int=0,
                delta:torch.Tensor|None = None,
                num_iter:int = 1)->None:
        if(delta is None):
            self.theta = torch.tensor([torch.pi],)
        else:
            self.theta =  torch.pi + delta
        self.target_state = target_state
        self.n = n
        self.N = 2**n
        self.build_oper()
        self.num_iter = num_iter
        self.ix = 1
        return
    
    def build_oper(self)->None:
        Ckz = CkiRz_torch(self.theta, self.n-1)
        Uf_ = Uf(self.target_state, self.n, Ckz, torch.tensor(sigma[1]))
        Uperp_ = Uperp(self.n, Ckz, torch.tensor(sigma[1]))
        self.O = torch.einsum('Aij, Ajk -> Aik',  Uperp_, Uf_)
        self.psi = uncertain_psi(self.n)
        self.psi = torch.einsum('Aij, j->Ai', self.O, self.psi)
        
        return 
    
    def __iter__(self):
        return self
    
    def __next__(self)->None:
        
        if(self.ix<self.num_iter):
            self.psi =  torch.einsum('Aij, Aj->Ai', self.O, self.psi)
            self.ix+=1
            return 
        else:
            raise StopIteration
    
    def __call__(self, psi):
        return torch.einsum('Aij, j->Ai', self.O, psi)
        
def uncertain_psi(n)->torch.Tensor:
    return torch.ones(2**n, dtype=torch.complex64)/torch.sqrt(torch.tensor(2**n))