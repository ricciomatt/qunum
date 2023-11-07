import torch 
from numpy.typing import NDArray
from . import representations

def commutator(A:torch.Tensor|NDArray, B:torch.Tensor|NDArray)->torch.Tensor|NDArray:
    return A @ B - B @ A

def anti_commutator(A:torch.Tensor|NDArray, B:torch.Tensor|NDArray)->torch.Tensor|NDArray:
    return A @ B + B @ A