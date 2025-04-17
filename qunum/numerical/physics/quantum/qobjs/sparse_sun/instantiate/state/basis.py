from ...core import SUNState
from torch import zeros, dtype as torchDtype, device as torchDevice, complex128, ones, tensor, rand

def init_sparse_basis(n_particles:int, suN:int = 2, oneStates:int = 0, objTp:str = 'ket', dtype:torchDtype= complex128, device:torchDevice = 'cpu', **kwargs)->SUNState:
    assert oneStates>0 and oneStates<suN**2, ValueError('oneStates must be an integer between \in (0, 1... suN**2({suN2})) but got {one}'.format(suN2=str(suN**2), one = oneStates))
    psi = zeros((1, n_particles, suN), dtype = dtype, device=device) 
    psi[:, oneStates, 0] = 1 + 0j
    return SUNState(
        psi, 
        coefs = ones(1, dtype=dtype, device=device),
        objTp=objTp,
        N = suN
    )

def uniform_superposition(n_particles:int, suN:int = 2, objTp:str = 'ket', dtype:torchDtype= complex128, device:torchDevice = 'cpu', **kwargs)->SUNState:
    '''\\ket{\\psi} = \\bigotimes_{n=0}^{N}(\\frac{1}{\sqrt{suN}} \sum_{n}\ket{n})'''
    psi = ones((1, n_particles, suN), dtype = dtype, device=device) 
    psi /= (psi*psi.conj()).sum(dim=-1).prod(dim=-1).pow(1/(2*psi.shape[1]))
    return SUNState(
        psi, 
        coefs = tensor([1+0j], dtype=dtype, device=device)/((tensor(2).pow(n_particles/2))),
        objTp = objTp,
        N = suN
    )
def random_product_state(n_particles:int, suN:int = 2, objTp:str = 'ket', dtype:torchDtype= complex128, device:torchDevice = 'cpu', **kwargs)->SUNState:
    psi = rand((1, n_particles, suN), dtype = dtype, device = device)
    psi/(psi[0,:]*psi[0,:].conj()).sum(dim = 0).pow(1/(2*psi.shape[1]))
    return SUNState(
        psi, 
        coefs = tensor([1+0j], dtype=dtype, device=device), 
        objTp = objTp,
        N = suN
    )
