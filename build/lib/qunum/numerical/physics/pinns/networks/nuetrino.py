from typing import Any
import torch
from torch.nn import Module, Linear, Sigmoid, Tanh, Softmax, Conv1d, Sequential, ReLU, LeakyReLU as LeLU
from torch import Tensor
from ....mathematics.algebra.representations import su
from ....mathematics.algebra import commutator as comm
from ...quantum.qobjs.torch_qobj import TQobj, direct_prod as dp_
from torch.linalg import matrix_exp as expm
from typing import Callable

class TimeEvolutionNuetrino:
    def __init__(self, 
                 H:Tensor,
                 num_layers:int,
                 dims:int = 2,)->None:
        
        
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
            
        )
        self.HBasis = H
        return
    
    def __call__(self, x:Tensor)->Tensor:
        R = self.GammaReal(x.real)
        I = self.GammaImag(x.real)
        Gamma = R + 1j*I
        U = self.HBasis[0] * Gamma[0]
        for h in range(1, self.HBasis.shape[0]):
            U += self.HBasis[h]*Gamma[h]
        return expm(U)
    
    def forward(self, x:Tensor)->Tensor:
        return self.__call__(x)
