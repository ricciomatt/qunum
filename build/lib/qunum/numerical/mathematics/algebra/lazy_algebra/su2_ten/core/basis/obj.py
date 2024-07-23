import torch 
from .core import doMul
from typing import Self

class SuBasis:
    def __init__(self, data:torch.Tensor, particle_inidicies:torch.Tensor|None = None)->Self:
        assert isinstance(data, torch.Tensor), TypeError('Must be Tensor type ')
        assert data.shape[-1] == 4, RuntimeError('Dimension indicies of this should be 4')
        self.data:torch.Tensor = data
        if(particle_inidicies is not None):
            self.particle_index:torch.Tensor = particle_inidicies.to(dtype = torch.int64)
        return
    
    def __matmul__(self, b:Self)->Self:
        return SuBasis(doMul(self.data, b.data))
    
    def isMatch(self, b:Self)->Self:
        return
    
    def __ee__(self, b):
        return