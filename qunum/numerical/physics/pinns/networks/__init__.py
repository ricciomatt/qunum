from typing import Any
import torch
from torch.nn import Module, Linear, Sigmoid, Tanh, Softmax, Conv1d, Sequential, ReLU, LeakyReLU as LeLU
from torch import Tensor
from ....mathematics.algebra.representations import su
from ...quantum.qobjs.torch_qobj import TQobj, direct_prod as dp_
from torch.linalg import matrix_exp as expm
from typing import Callable



class GeneralTimeEvolutionNN:
    def __init__(self, 
                 n_particles:int, 
                 HamiltonianBasis:Tensor,
                 dims:int = 2,)->None:
        self.real = Sequential(
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
        self.imag = Sequential(
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
        
        
        if(dims == 2):
            self.sigma = TQobj(su.get_pauli(to_tensor=True), n_particles=1)
        elif(dims == 3):
            self.sigma = TQobj(su.get_gellmann(to_tensor=True), n_particles=1, hilbert_space_dims=3)    
        else:
            raise ValueError('Dims must be 2 or 3')
        self.sigma = self.sigma.to_tensor()
        return
    
    def __call__(self, x:Tensor)->Tensor:
        R = self.real(x.real)
        I = self.imag(x.imag)
        Gamma = R + 1j*I
        return expm(torch.einsum('Ai, imn -> Amn',Gamma, self.sigma))

    
    def forward(self, x:Tensor)->Tensor:
        return self.__call__(x)
