import torch
from torch import Tensor

@torch.jit.script
def reg(X:Tensor, Y:Tensor,lam:float)->Tensor:
    XtXi = torch.linalg.inv(X.T @ X)
    return XtXi @ X.T @ Y