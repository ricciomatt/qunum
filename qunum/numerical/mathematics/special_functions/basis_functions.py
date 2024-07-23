import torch
from typing import Callable, Self, Any

@torch.jit.script
def dirac_delta(x:torch.Tensor)->torch.Tensor:
    return (x==torch.zeros(1, dtype = x.dtype)[0]).to(x.dtype)

@torch.jit.script
def heavside_theta(x:torch.Tensor)->torch.Tensor:
    return (x>=torch.zeros(1, dtype = x.dtype)[0]).to(x.dtype)

@torch.jit.script
def gaussian_delta(x:torch.Tensor, sigma:float = 1e-8)->torch.Tensor:
    return (-(x**2)/(2*sigma)).exp()
