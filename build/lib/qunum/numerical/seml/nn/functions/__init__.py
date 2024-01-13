import torch
from torch import Tensor
class Cos:
    def __init__(self):
        return
    def __call__(self, x:Tensor) -> Tensor:
        return torch.cos(x)

class Sin:
    def __init__(self):
        return
    def __call__(self, x:Tensor) -> Tensor:
        return torch.sin(x)
    
class Expm:
    def __init__(self):
        return 
    def __call__(self, x:Tensor)->Tensor:
        return torch.linalg.matrix_exp(x) 