import torch
from torch import einsum

def lagrange_interpolation(f:torch.Tensor, x:torch.Tensor, num_pts:int= int(1e3))->torch.Tensor:
    phi = torch.linspace(float(x.min()), float(x.max()), num_pts,)
    Phi_jk = (einsum('j, k-> jk', x, torch.ones_like(x)) - x)
    Phi_ik = einsum('i, k -> ik', phi, torch.ones_like(x))- x
    A = lagrange_prod(Phi_ik, Phi_jk)
    return einsum('ij, j -> i', A, f), phi

   
@torch.jit.script 
def lagrange_prod(Phi_ik, Phi_jk):
    A = torch.ones((Phi_ik.shape[0],Phi_jk.shape[0]))
    for j in range(A.shape[1]):
        for k in range(A.shape[1]):
            if(j!=k):
                A[:,j] *= Phi_ik[:,k]/(Phi_jk[j,k])
    return A



def lagrange_interp_coef(x:torch.Tensor, num_pts = int(1e2))->torch.Tensor:
    phi = torch.linspace(float(x.min()), float(x.max()), num_pts,)
    Phi_jk = (einsum('j, k-> jk', x, torch.ones_like(x)) - x)
    Phi_ik = einsum('i, k -> ik', phi, torch.ones_like(x))- x
    A = lagrange_prod(Phi_ik,Phi_jk)
    return ((A[0:A.shape[0]-1]+A[1:])/2).sum(dim=0)*(phi[1]-phi[0])
    

