from typing import Callable, Any
import torch

def pims_torch(y:torch.tensor, yh:torch.tensor, **kwargs)->torch.tensor:
    pass 

def bianry_cross_entropy(y:torch.tensor, yh:torch.tensor, **kwargs)->torch.tensor:
    return(-1*(y*torch.log(torch.abs(yh-1e-8)) + (1-y)*torch.log(torch.abs(1-yh+1e-8)))).sum()

def least_squares(y:torch.tensor, yh:torch.tensor, **kwargs)->torch.tensor:
    return torch.pow((y - yh),2).sum()

def least_squares_encoder(y:torch.tensor, yh:torch.tensor, **kwargs)->torch.tensor:
    return torch.pow((y - yh),2).sum()

def cat_cross_entropy(y:torch.tensor,yh:torch.tensor, **kwargs)->torch.tensor:
    return(-y*torch.log(torch.abs(yh-1e-8))).sum()


