
# Module Level Imports from Torch
from torch.nn import Sequential, Module
# loss Level Imports From torch
from torch.nn import MSELoss, CrossEntropyLoss, KLDivLoss
# Layer Level imports from torch
from torch.nn import Linear, Conv2d, ConvTranspose2d, BatchNorm2d, AvgPool2d, MaxPool2d, Flatten
#Activation functions from Torch
from torch.nn import Softmax, Sigmoid, ReLU, LeakyReLU, Softmax2d
#Operations from torch
from torch import tensor, Size, Tensor
#distributions from torch
from torch.distributions import Normal

from ... import loss_functions as lf
from typing import Callable
import numpy as np

class SimpleCNN(Module):
    def __init__(self, 
                 out_shp:int = 9,
                 loss_funct:Callable =  CrossEntropyLoss(),
                 )->object:
        super(SimpleCNN, self).__init__()
        self.conv_layers = Sequential(
            Conv2d(3, 16, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(16, eps=1e-3),
            MaxPool2d((2,2)),
            
            Conv2d(16, 32, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(32, eps=1e-3),
            MaxPool2d((2,2)),
        )
        self.flat = Flatten(1)
        self.linear_layers = Sequential(
            Linear(5408, 2000),
            LeakyReLU(),
            
            Linear(2000, out_shp),
            Softmax(dim=1),
            
        )
        self.loss_funct = loss_funct
        
    def __call__(self, x):
        return self.forward(x)    
    
    def forward(self, x):
        return self.linear_layers(self.flat(self.conv_layers(x)))

    
    def eval_loss(self, x, y):
        return self.loss_funct(y, self.forward(x))
        
class UnsupervisedCNN(Module):
    def __init__(self, 
                 inp_shape:int = (3,50,50),
                 num_classes:int = int(9),
                 loss_funct:Callable =  lf.ul.UnsupOdds(),
                 )->object:
        super(SimpleCNN, self).__init__()
        self.conv_layers = Sequential(
            Conv2d(inp_shape[0], 16, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(16, eps=1e-3),
            MaxPool2d((2,2)),
            
            Conv2d(16, 32, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(32, eps=1e-3),
            MaxPool2d((2,2)),
        )
        self.flat = Flatten(1)
        self.linear_layers = Sequential(
            Linear(5408, 2000),
            LeakyReLU(),
            
            Linear(2000, num_classes),
            Softmax(dim=1),
            
            
        )
        self.loss_funct = loss_funct
        
    def __call__(self, x):
        return self.forward(x)    
    
    def forward(self, x):
        return self.linear_layers(self.flat(self.conv_layers(x)))
    
class UnsupervisedClassifier(Module):
    def __init__(self, 
                 inp_shape:int = (3,50,50),
                 num_classes:int = 9,
                 loss_funct:Callable =  lf.ul.UnsupOdds(),
                 )->classmethod:
        super(SimpleCNN, self).__init__()
        self.conv_layers = Sequential(
            Conv2d(inp_shape[0], 16, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(16, eps=1e-3),
            MaxPool2d((2,2)),
            
            Conv2d(16, 32, (2,2), padding = 1),
            LeakyReLU(),
            BatchNorm2d(32, eps=1e-3),
            MaxPool2d((2,2)),
        )
        self.flat = Flatten(1)
        self.linear_layers = Sequential(
            Linear(5408, 2000),
            LeakyReLU(),
            
            Linear(2000, num_classes),
            Softmax(dim=1),
            
            
        )
        self.loss_funct = loss_funct
        
    def __call__(self, x):
        return self.forward(x)    
    
    def forward(self, x):
        return self.linear_layers(self.flat(self.conv_layers(x)))
