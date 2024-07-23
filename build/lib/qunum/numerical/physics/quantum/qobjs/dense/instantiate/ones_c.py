from ..core.torch_qobj import TQobj
import numpy as np 
from typing import Iterable
import torch
from .core import extract_shape

def ones(shp:int|Iterable[int], dtype:torch.TypedStorage = torch.complex128, device:str|int = 'cpu', **kwargs)->TQobj:
    return TQobj(torch.ones(shp,  device= device, dtype= dtype), **kwargs)

def ones_like(a:TQobj)->TQobj:
    assert isinstance(a, TQobj), 'Must be a TQobj'
    b:TQobj  = torch.zeros_like(a)
    b.set_meta(a._metadata)
    return b

def ones_from_context(shp:int|list[int]|None = None, n_particles:int = 1, hilbert_space_dims:int = 2, dims:dict[int:int]|None = None, obj_tp:str = 'ket', **kwargs)->TQobj:
    if(dims is not None):
        l = np.prod(list(dims.values()))    
    else:
        l = hilbert_space_dims**n_particles
    return ones(
        extract_shape(obj_tp, l, shp), 
        n_particles=n_particles, 
        dims=dims, 
        hilbert_space_dims=hilbert_space_dims, 
        obj_tp=obj_tp, 
        **kwargs
    )
