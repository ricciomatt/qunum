import torch
from typing import Callable
from torch import Tensor
class LinearNN(torch.nn.Module):
    def __init__(self, inpt_dims:int, out_dims:int, F:Callable):
        super(LinearNN, self).__init__()
        self.linear = torch.nn.Linear(inpt_dims, out_dims, bias= False)
        self.F = F        
        return
    def forward(self, x:Tensor)->Tensor:
        return self.F(self.linear(x))
    
    def __call__(self, x:Tensor)->Tensor:
        return self.forward(x)
    
