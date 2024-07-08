import torch
from ...qobjs import TQobj
import numpy as np

#@torch.jit.script
def pmns2(theta:torch.Tensor=torch.tensor(torch.pi/5), dtype = torch.complex128):
    return TQobj(torch.tensor([[torch.cos(theta), torch.sin(theta)],[-torch.sin(theta), torch.cos(theta)]], dtype=dtype), n_particles=1, hilbert_space_dims=2)

def pmns3(
        theta12:torch.Tensor=torch.tensor(torch.pi/5), 
        theta13:torch.Tensor=torch.tensor(torch.pi/5),
        theta23:torch.Tensor=torch.tensor(torch.pi/5), 
        deltacp:torch.Tensor=torch.tensor(0),
        dtype = torch.complex128
    )->TQobj:
    return (
        TQobj(
            torch.tensor([
                [1,0,0],[0,torch.cos(theta23), torch.sin(theta23)],
                [0, -torch.sin(theta23), torch.cos(theta23)]
                ],dtype=dtype), 
            n_particles=1, 
            hilbert_space_dims=3)
        @TQobj(
            torch.tensor([
                [torch.cos(theta13),0, torch.sin(theta13)*torch.exp(-1j*deltacp)],
                [0,1,0],
                [-torch.sin(theta13)*torch.exp(1j*deltacp), 0, torch.cos(theta13)]
                ], dtype=dtype), 
            n_particles=1, 
            hilbert_space_dims=3
        )@TQobj(
            torch.tensor([
                [torch.cos(theta12), torch.sin(theta12),0],
                [-torch.sin(theta12), torch.cos(theta12),0],
                [0,0,1]], 
                dtype=dtype), 
            n_particles=1, 
            hilbert_space_dims=3
        )
    )

