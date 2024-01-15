import torch
from torch import Tensor

@torch.jit.script
def ridge_reg(X:Tensor, Y:Tensor, lam:float)->Tensor:
    XtX = X.T @ X 
    XtX -= lam * torch.eye(XtX.shape[0], XtX.shape[1])
    XtXi = torch.linalg.inv(XtX)    
    return XtXi @ X.T @ Y
