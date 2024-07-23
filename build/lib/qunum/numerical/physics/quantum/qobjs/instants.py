from .torch_qobj import TQobj
import torch

def eye(dim:int, dtype:torch.TypedStorage = torch.complex128, device:str|int = 'cpu', **kwargs)->TQobj:
    return TQobj(torch.eye(dim,dtype=dtype).to(device=device), **kwargs)

def zeros(dims:tuple, dtype:torch.TypedStorage = torch.complex128, device:str|int = 'cpu', **kwargs)->TQobj:
    return TQobj(torch.zeros(dims,  device= device, dtype= dtype), **kwargs)

def zeros_like(a:TQobj)->TQobj:
    assert isinstance(a, TQobj), 'Must be a TQobj'
    return torch.zeros_like(a).set_meta(a._metadata)
