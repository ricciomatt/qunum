import torch 
from numpy.typing import NDArray
from .representations import *

def commutator(A:torch.Tensor|NDArray, B:torch.Tensor|NDArray)->torch.Tensor|NDArray:
    return A @ B - B @ A

def anti_commutator(A:torch.Tensor|NDArray, B:torch.Tensor|NDArray)->torch.Tensor|NDArray:
    return A @ B + B @ A

def dexp(A):
    pass
@torch.jit.script
def tensor_commutator(A:torch.Tensor, B:torch.Tensor)->torch.Tensor:
    return A @ B - B @ A

@torch.jit.script
def ad(A:torch.Tensor, B:torch.Tensor, k:int = 1)->torch.Tensor:
    for i in range(k+1):
        t = B.clone()
        B = tensor_commutator(A,t)
    return B
    