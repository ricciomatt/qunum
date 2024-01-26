import torch
from ...qobjs import TQobj
import numpy as np

#@torch.jit.script
def pmns2(theta:torch.Tensor=torch.pi/5, dtype = torch.complex128):
    return TQobj(torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]], dtype=dtype), n_particles=1, hilbert_space_dims=2)
