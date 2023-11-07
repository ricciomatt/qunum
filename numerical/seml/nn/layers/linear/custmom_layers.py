from typing import Any, Callable
from torch import Tensor, zeros, pow, exp
from torch.nn import Parameter, Module
from torch.nn.init import kaiming_uniform_, uniform_
import numpy as np
import torch

class PartitionExpectation(Module):
    def __init__(self, 
                 size_in:int, 
                 num_states:int, 
                 size_out:int ) -> object:
        """_summary_

        Args:
            size_in (int): _description_
            num_states (int): _description_
            size_out (int): _description_

        Returns:
            object: _description_
        """        
        super(PartitionExpectation, self).__init__()
        state = Tensor(num_states, size_in)
        self.state = Parameter(state)  # nn.Parameter is a Tensor that's a module parameter.
        
        state_spacing = Tensor(num_states, size_in)
        self.state_spacing = Parameter(state_spacing)

        state_vals = Tensor(size_out, num_states)
        self.state_vals  = Parameter(state_vals)
        
        kaiming_uniform_(self.state, a=np.sqrt(5)) # weight init
        uniform_(self.state_spacing, a=0, b=1) # weight init
        uniform_(self.state_vals, a=0, b = 1) # weight init
        return
    
    def __call__(self, x:Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x:Tensor) -> Tensor:
        z = zeros(x.shape[0],self.state.shape[0])
        for i in range(x.shape[1]):
            z +=  pow((x[:,i].reshape(-1, 1) - self.state[:,i]), 2)*self.state_spacing[:, i]
        z = exp(-z)
        return (z/z.sum(0))@self.state_vals.T


class HadamardLayer(Module):
    def __init__(self,
                 size_in:tuple[int|int,int|int,int,int])->object: 
        """_summary_

        Args:
            size_in (tuple[int | int,int | int,int,int]): _description_

        Returns:
            object: _description_
        """               
        super(HadamardLayer, self).__init__()
        print(size_in)
        self.w = Parameter(Tensor(size = size_in))
        self.b = Parameter(Tensor(size = size_in))
        print(self.w.shape)
        
        kaiming_uniform_(self.w, a=np.sqrt(5)) # weight init
        uniform_(self.b)
        
    def __call__(self, x:Tensor)->Tensor:
        return self.forward(x)
    
    def forward(self, x:Tensor)->Tensor:
        return x*self.w + self.b


