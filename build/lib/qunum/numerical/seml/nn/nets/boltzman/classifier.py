# Module Level Imports from Torch
from torch.nn import Sequential, Module
# loss Level Imports From torch
from torch.nn import CrossEntropyLoss, CosineSimilarity, KLDivLoss

# Layer Level imports from torch

from torch.nn import Linear
#Activation functions from Torch
from torch.nn import Softmax, Sigmoid, ReLU, LeakyReLU
#Operations from torch
from torch import tensor, Size, Tensor
#distributions from torch
from torch.distributions import Normal

from ...loss_functions import bianry_cross_entropy, cat_cross_entropy
from typing import Callable
import numpy as np




class LinearClassifierFromShape(Module):
    def __init__(self, 
                 inp_shape:int,
                 out_shape:int,
                 funct_out:Callable = Softmax(1), 
                 loss_funct:Callable = bianry_cross_entropy):
        super(LinearClassifier, self).__init__()
        if(funct_out is None):
            funct_out = Softmax(dim =1)
        self.funct_out = funct_out
        self.classify = Sequential(
            Linear(inp_shape, inp_shape*2),
            ReLU(),
            
            Linear(inp_shape*2, inp_shape),
            ReLU(),
            
            Linear(inp_shape, out_shape),
            funct_out
        )
        self.loss_funct = loss_funct
        
    def __call__(self, x:Tensor)->Tensor:
        return self.forward(x)
    
    def forward(self,x:Tensor)->Tensor:
        return self.classify(x)



class LinearClassifier(Module):
    def __init__(self, 
                 classifier:Callable, 
                 loss_funct:Callable = bianry_cross_entropy):
        super(LinearClassifier, self).__init__()
        self.classify = classifier
        self.loss_funct = loss_funct
        self.use = self.classify
        
    def __call__(self, x:Tensor)->Tensor:
        return self.forward(x)
    
    def forward(self,x:Tensor)->Tensor:
        return self.classify(x)


