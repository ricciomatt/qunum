import torch
from ..core.torch_qobj import TQobj
from torch import swapaxes as SW
def rand_hermitian(
        size:tuple, 
        dist:None|list[torch.distributions.Distribution,torch.distributions.Distribution] = None, 
        *args:tuple[int], 
        to_tensor:bool = False, 
        dtype:torch.dtype = torch.complex128, 
        n_particles:int=1, 
        hilbert_space_dims:int= 2, 
        dims:dict|None =None,  
        **kwargs
    )->TQobj|torch.Tensor:
    assert size[-1] == size[-2], ValueError('Must be a Symmetric Matrix')
    if(dist is None):
        dist = [torch.distributions.Uniform(0,1),torch.distributions.Uniform(0,1)]
    T = torch.complex(dist[0].rsample(size), dist[1].rsample(size))
    T = T + SW(T.conj(),-1,-2)
    if(to_tensor):
        return T.to(dtype)
    else:
        return TQobj(T, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dtype=dtype, dims = dims)

def rand_unitary(
        size:tuple, 
        dist:None|list[torch.distributions.Distribution,torch.distributions.Distribution] = None, 
        *args:tuple[int], 
        to_tensor:bool = False, 
        dtype:torch.dtype = torch.complex128, 
        n_particles:int=1, 
        hilbert_space_dims:int= 2, 
        dims:dict|None =None,  
        **kwargs
    )->TQobj|torch.Tensor:
    assert size[-1] == size[-2], ValueError('Must be a Symmetric Matrix')
    if(dist is None):
        dist = [torch.distributions.Uniform(0,1),torch.distributions.Uniform(0,1)]
    T = torch.complex(dist[0].rsample(size), dist[1].rsample(size))
    T, Q = torch.linalg.qr(T) 
    if(to_tensor):
        return T.to(dtype)
    else:
        return TQobj(T, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dims = dims, dtype= dtype)

def rand_state(
        number_pts:int = 1, 
        n_particles:int=1, 
        hilbert_space_dims:int= 2, 
        dims:dict|None =None, 
        dtype:torch.dtype = torch.complex128, 
        to_density:bool = False, 
        **kwargs
    )->TQobj:
    if(dims is None):
        l = hilbert_space_dims**n_particles
        args = (number_pts, l, 1)
    else:
        l = 1
        for d in dims:
            l+=d**dims[d]
        args = (number_pts, l, 1)
    
    S = TQobj(torch.rand(args, dtype=dtype), n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dtype=dtype, dims = dims, **kwargs)
    with torch.no_grad():
        S/=torch.sqrt(S.dag() @ S)
    if(to_density):
        return S.to_density()
    else:
        return S