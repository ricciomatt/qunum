import torch
from ....physics.quantum.qobjs.torch_qobj import TQobj
from torch import swapaxes as SW
def rand_hermitian(size:tuple, dist:None|list[torch.distributions.Distribution,torch.distributions.Distribution] = None, *args:tuple[int], to_tensor:bool = False, dtype:torch.TypedStorage = torch.complex128, n_particles:int=1, hilbert_space_dims:int= 2, dims:dict|None =None,  **kwargs)->TQobj|torch.Tensor:
    assert size[-1] == size[-2], ValueError('Must be a Symmetric Matrix')
    if(dist is None):
        dist = [torch.distributions.Uniform(0,1),torch.distributions.Uniform(0,1)]
    T = torch.complex(dist[0].rsample(size), dist[1].rsample(size))
    
    T = T + SW(T.conj(),-1,-2)
    if(to_tensor):
        return T.to(dtype)
    else:
        return TQobj(T, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dtype=dtype, dims = dims)

def rand_unitary(size:tuple, dist:None|list[torch.distributions.Distribution,torch.distributions.Distribution] = None, *args:tuple[int], to_tensor:bool = False, dtype:torch.TypedStorage = torch.complex128, n_particles:int=1, hilbert_space_dims:int= 2, dims:dict|None =None,  **kwargs)->TQobj|torch.Tensor:
    assert size[-1] == size[-2], ValueError('Must be a Symmetric Matrix')
    if(dist is None):
        dist = [torch.distributions.Uniform(0,1),torch.distributions.Uniform(0,1)]
    T = torch.complex(dist[0].rsample(size), dist[1].rsample(size))
    T, Q = torch.linalg.qr(T) 
    if(to_tensor):
        return T.to(dtype)
    else:
        return TQobj(T, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dims = dims, dtype= dtype)

def rand_state(*args, number_pts:int = 1,  n_particles:int=1, hilbert_space_dims:int= 2, dims:dict|None =None, dtype:torch.TypedStorage = torch.complex128, to_density:bool = False, **kwargs)->TQobj:
    if(dims is None):
        l = n_particles**hilbert_space_dims
    else:
        l = 1
        for d in dims:
            l+=d**dims[d]
    S = TQobj(torch.rand((number_pts, l, 1),*args, dtype=dtype,**kwargs), n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dtype=dtype, dims = dims)
    with torch.no_grad():
        S/=torch.sqrt(S.dag() @ S)
    if(to_density):
        return S.to_density()
    else:
        return S