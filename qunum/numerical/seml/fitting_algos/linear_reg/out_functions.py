import numpy as np
import torch
from typing import Callable
def getMap(use_numpy:bool = False)->dict[str:Callable|object]:
    return {
        "None":Id, 
        "sigmoid":Sigmoid, 
        'tanh':Tanh,
        'tanh_prb':PrbTanh,
        "sin":Sin,
        "cos":Cos,
        "exp":Exp,
        'cosh':Cosh,
        'sinh':Sinh,
        'sqrt':Sqrt,
        'pow':Pow,
        'to_pow':ToPow,
    } 

class Id:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return x
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return x

class Sigmoid:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return 1/(1+np.exp(x))
        else:
            return 1/(1+torch.exp(x))
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.log(1/x-1)
        else:
            return 1/(1+torch.exp(x))

class Tanh:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.tanh(x)
        else:
            return torch.tanh(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arctanh(x)
        else:
            return torch.arctanh(x)

class PrbTanh:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return (np.tanh(x)+1)/2
        else:
            return (torch.tanh(x)+1)/2
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arctanh(2*x-1)
        else:
            return torch.arctanh(2*x-1)


class Exp:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.exp(x)
        else:
            return torch.exp(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.log(x)
        else:
            return torch.log(x)

class Cosh:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.cosh(x)
        else:
            return torch.cosh(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arccosh(x)
        else:
            return torch.arccosh(x)


class Sinh:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.sinh(x)
        else:
            return torch.sinh(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arcsinh(x)
        else:
            return torch.arcsinh(x)


class Cos:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.cos(x)
        else:
            return torch.cos(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arccos(x)
        else:
            return torch.arccos(x)


class Sin:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.sin(x)
        else:
            return torch.sin(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.arcsin(x)
        else:
            return torch.arcsin(x)

class Sqrt:
    def __init__(self, use_numpy:bool=False, **kwargs)->None:
        self.use_numpy = use_numpy
        return
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.sqrt(x)
        else:
            return torch.sqrt(x)
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return x**2

class Pow:
    def __init__(self,*args, use_numpy:bool=False, **kwargs):
        self.power = kwargs['power']
        self.use_numpy = use_numpy
        return 
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return x**self.power
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return x**(1/self.power)

class ToPow:
    def __init__(self,*args, use_numpy:bool=False, **kwargs):
        self.base = kwargs['base']
        self.use_numpy = use_numpy
        return 
    def __call__(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        return self.base**x
    def inv(self, x:torch.Tensor|np.ndarray)->torch.Tensor|np.ndarray:
        if(self.use_numpy):
            return np.log(x)/np.log(self.base)
        else:
            return torch.log(x)/torch.log(self.base)