from ...core import PauliState
from torch import zeros, dtype as torchDtype, device as torchDevice, complex128, ones, tensor

def mk_sparse_basis(n_particles:int, oneStates:None|tuple[int] = None, objTp:str = 'ket', dtype:torchDtype= complex128, device:torchDevice = 'cpu', **kwargs)->PauliState:
    psi = zeros((1, n_particles, 2), dtype = dtype, device=device) 
    psi[0,:,0] = 1 + 0j
    if(oneStates is not None):
        psi[0, oneStates, 0] = 0 + 0j
        psi[0, oneStates, 1] = 1 + 0j
    return PauliState(
        psi, 
        coefs = ones(1, dtype=dtype, device=device),
        objTp=objTp
    )

def uniform_superposition(n_particles:int, oneStates:None|tuple[int] = None, objTp:str = 'ket', dtype:torchDtype= complex128, device:torchDevice = 'cpu', **kwargs)->PauliState:
    psi = ones((1,n_particles, 2), dtype = dtype, device=device) 
    return PauliState(
        psi, 
        coefs = tensor([1+0j], dtype=dtype, device=device)/((tensor(2).pow(n_particles/2))),
        objTp=objTp
    )