from typing import Any
import torch
from torch.nn import Module, Linear, Sigmoid, Tanh, Softmax, Conv1d, Sequential, ReLU, LeakyReLU as LeLU
from torch import Tensor
from torch.linalg import matrix_exp as expm
from typing import Callable
from ....mathematics import einsum
class TimeEvolutionNuetrino(Module):
    def __init__(self, 
                 H:Tensor
                 )->None:
        super(TimeEvolutionNuetrino, self).__init__()
        self.GammaReal = Sequential(
            Linear(1, 48),
            LeLU(),
            
            Linear(48, 128),
            LeLU(),
            
            Linear(128, 256),
            LeLU(),
            
            Linear(256, 512),
            LeLU(),
            
            Linear(512, 256),
            LeLU(),
            
            Linear(256, 128),
            LeLU(),
            
            Linear(128, 48),
            LeLU(),
            
            Linear(48, H.shape[0])
            
        )
        self.GammaImag = Sequential(
            Linear(1, 48),
            LeLU(),
            
            Linear(48, 128),
            LeLU(),
            
            Linear(128, 256),
            LeLU(),
            
            Linear(256, 512),
            LeLU(),
            
            Linear(512, 256),
            LeLU(),
            
            Linear(256, 128),
            LeLU(),
            
            Linear(128, 48),
            LeLU(),
            
            Linear(48, H.shape[0])
            
        )
        self.HBasis = H
        return
    
    def __call__(self, x:Tensor)->Tensor:
        R = self.GammaReal(x.real)
        I = self.GammaImag(x.real)
        Gamma = torch.complex(R, I)
        return (einsum('bij, Ab->Aij', self.HBasis, Gamma)).expm()
    
    def forward(self, x:Tensor)->Tensor:
        return self.__call__(x)
