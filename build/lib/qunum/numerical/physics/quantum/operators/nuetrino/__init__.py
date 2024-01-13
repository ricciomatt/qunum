import torch
from ...qobjs import TQobj
import numpy as np

#@torch.jit.script
def pmns2(theta:torch.Tensor):
    return TQobj(torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]], dtype=torch.complex128), n_particles=1, hilbert_space_dims=2)
