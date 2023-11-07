# Module Level Imports from Torch
from torch.nn import Sequential, Module
# loss Level Imports From torch
from torch.nn import MSELoss
# Layer Level imports from torch
from torch.nn import Linear, Conv2d, ConvTranspose2d, BatchNorm2d, AvgPool2d, MaxPool2d, Flatten, Unflatten
#Activation functions from Torch
from torch.nn import Softmax, Sigmoid, ReLU, LeakyReLU, Softmax2d
#Operations from torch
from torch import tensor, Size, Tensor, rand
#distributions from torch
from torch.distributions import Normal

from numpy import prod

from ...loss_functions import *
from typing import Callable
import numpy as np
from ...layers import image_layers as iml
class VAE(Module):
    def __init__(self, 
                 ishp:tuple[int|int,int|int,int,int] = (3, 42, 42),
                 latent_dims:int = 64,
                 loss_funct:Callable = MSELoss(), 
                 )->object:
        super(VAE, self).__init__()
        self.conv_encoder = Sequential(
            #Convolutional Layers
            Conv2d(ishp[0], 16, 2, stride = 2, padding = 1), 
            BatchNorm2d(16),
            LeakyReLU(),
            Conv2d(16, 32, 2, stride = 2, padding = 1),
            LeakyReLU(),  
            Conv2d(32, 42, 2, stride=2,),  
            LeakyReLU(),
                                    
        )
        sp = [1]
        sp.extend(ishp)
        b = self.conv_encoder(rand(sp)).shape
        
        self.linear_encoder = Sequential(
            Flatten(1),
            Linear(prod(b), 128),
            LeakyReLU(),
        )
        
        
        self.mu = Sequential(
            Linear(128, latent_dims),
            ReLU(),
        )
        self.sig = Sequential(
            Linear(128, latent_dims),
            ReLU(),
            
        )
        self.linear_decoder = Sequential(
            Linear(latent_dims, 128),
            ReLU(True),    
            Linear(128, prod(b)),
            ReLU()
        )

        self.conv_decoder = Sequential(
            Unflatten(dim = 1, unflattened_size=tuple(b[1:])),
            ConvTranspose2d(42, 32, 2, stride = 2),
            LeakyReLU(),
            ConvTranspose2d(32, 16, 2, stride = 2, padding = 1 ), 
            ReLU(),
            ConvTranspose2d(16, 3, 2, stride=2, padding = 1),
            Softmax2d(),
            
        )
        
        self.pdf = Normal(0, 1)
        
        self.latent_dims = latent_dims 
        
        self.loss_funct = loss_funct
        
        
    def encoder(self, x:Tensor)->Tensor:
        x = self.conv_encoder(x) 
        x = self.linear_encoder(x)
        x = self.mu(x) + self.sig(x)*self.pdf.sample((x.shape[0], self.latent_dims))
        return x
    
    def __call__(self, x:Tensor)->Tensor:
        return self.encoder(x)
    
    def decoder(self, x:Tensor)->Tensor:
        return self.conv_decoder(self.linear_decoder(x))
    
    def forward(self, x:Tensor)->Tensor:
        x = self.encoder(x)
        return self.decoder(x)
        