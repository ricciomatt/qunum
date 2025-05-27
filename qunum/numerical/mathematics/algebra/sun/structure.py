from .representations import su_n_generate
import torch
class SUAlgebra:
    def __init__(self, n:int = 3 ):
        T = su_n_generate(n, include_identity = False, ret_type='tensor')
        self.n = n
        self.f:torch.Tensor =  -1j*torch.einsum('ABij, Cji->ABC', torch.einsum('Aij, Bjk-> ABik', T,T) -torch.einsum('Aij, Bjk-> BAik', T,T), T)/2
        self.d:torch.Tensor = torch.einsum('ABij, Cji->ABC', torch.einsum('Aij, Bjk-> ABik', T,T) +torch.einsum('Aij, Bjk-> BAik', T,T), T)/2
        return