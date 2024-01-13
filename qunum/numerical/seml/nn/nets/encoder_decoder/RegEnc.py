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

class AutoEncoderFromShape(Module):
    def __init__(self, 
                 inp_shape:tuple, 
                 out_shape:tuple = (8,6,6),
                 loss_funct:Callable = least_squares_encoder, 
                 encoder:Sequential = None, 
                 decoder:Sequential = None, )->object:
        super(AutoEncoderFromShape, self).__init__()    
        self.encoder_layers = encoder    
        self.decoder_layers = decoder
        self.loss_funct = loss_funct
        self.use = self.encode
        
    def __call__(self, x:Tensor)->Tensor:
        return self.encode(x)
    
    def encode(self, x:Tensor)->Tensor:
        return self.encoder_layers(x)
    
    def decode(self, x:Tensor)->Tensor:
        return self.decoder_layers(x)
    
    def forward(self,x:Tensor)->Tensor:
        return self.decode(self.encode(x))



class AutoEncoder(Module):
    def __init__(self,
                 loss_funct:Callable = MSELoss(), 
                 encoder:Sequential = None, 
                 decoder:Sequential = None, )->object:
        super(AutoEncoder, self).__init__()
        self.encoder_layers = encoder    
        self.decoder_layers = decoder
        self.loss_funct = loss_funct
        
    def encode(self, x:Tensor)->Tensor:
        return self.encoder_layers(x)
    
    def decode(self, x:Tensor)->Tensor:
        return self.decoder_layers(x)
    
    def __call__(self, x:Tensor)->Tensor:
        return self.encode(x)
    
    def forward(self,x:Tensor )->Tensor:
        return self.decode(self.encode(x))
    
    def eval_loss(self, x:Tensor, y:Tensor)->Tensor:
        return self.loss_funct(y, self.forward(x))

    
class DenoisingEncoder(Module):
    def __init__(self,
                 inp_shape:tuple= (3,50,50),
                 centering_:float = .5, 
                 sigma:float = .5, 
                 loss_funct:Callable = MSELoss(), 
                 )->object:
        """_summary_

        Args:
            inp_shape (tuple, optional): _description_. Defaults to (3,50,50).
            centering_ (float, optional): _description_. Defaults to .5.
            sigma (float, optional): _description_. Defaults to .5.
            loss_funct (Callable, optional): _description_. Defaults to MSELoss().

        Returns:
            object: _description_
        """        
        
        super(DenoisingEncoder, self).__init__()
        
        self.normal_dist = Normal(centering_, sigma)
      
        self.encoder = Sequential(
            
            Conv2d(inp_shape[0], 16, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            ReLU(),
            
            Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1)),
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            LeakyReLU(negative_slope=0.01),
            
            Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1)),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            LeakyReLU(negative_slope=0.01),
            )
        self.decoder = Sequential(
            
            ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(1, 1)),
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(),
            
            ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(1, 1)),
            BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            LeakyReLU(negative_slope=0.01),
            
            ConvTranspose2d(16, inp_shape[0], kernel_size=(2, 2), stride=(2, 2)),
            BatchNorm2d(inp_shape[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            Softmax2d(),
            
            )
        print(self.decoder)
       
        self.loss_funct = loss_funct
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        if(self.training):
            x += self.normal_dist.sample(sample_shape=Size(x.shape)).to(x.device)
        return self.decoder(self.encoder(x))



        

        
        