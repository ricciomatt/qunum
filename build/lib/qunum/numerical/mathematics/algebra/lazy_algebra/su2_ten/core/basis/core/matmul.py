import torch 
from .......tensors import levi_cevita_tensor
from string import ascii_uppercase, ascii_lowercase
def doMul(a:torch.Tensor, b:torch.Tensor, e:torch.Tensor)->torch.Tensor:
    return mul_basis(a, b.to(device=a.device), levi_cevita_tensor(4, to_tensor=True).to(dtype = a.dtype, device= a.device), pfx =ascii_uppercase()[:len(a.shape)-1] )


@torch.jit.script
def mul_basis(a:torch.Tensor, b:torch.Tensor, e:torch.Tensor, pfx:str)->torch.Tensor:
    m = torch.zeros_like(a)
    for i in range(4):
        m[..., 0] += a[..., i]*b[...,i]
        m[:, i] += a[...,0]*b[...,i] + a[...,0]*b[...,i]
    m[..., 1:] += torch.einsum(pfx+'i,'+pfx+'j, ijk->'+pfx+'k', a, b, e)
    return m
        
