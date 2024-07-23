
from ..core.torch_qobj import TQobj
from torch import complex128, dtype as TorchDtype, eye as teye
from numpy import array
def eye(n_particles:int=1, hilbert_space_dims:int = 2, dims:dict[int:int]|None = None, dtype:TorchDtype = complex128, **kwargs)->TQobj:
    if(dims is not None):
        l = array(list(dims.values())).prod()
    else:
        l = hilbert_space_dims**n_particles
    return TQobj(
        teye(
            l, dtype = dtype
        ),
        dims=dims,
        n_particles=n_particles,
        hilbert_space_dims=hilbert_space_dims,
        **kwargs
    )

def eye_like(A:TQobj, **kwargs)->TQobj:
    return TQobj(
        teye(
            n=A.shape[-2], 
            m=A.shape[-1], 
            dtype = A.dtype, 
            device = A.device,
            requires_grad=A.requires_grad
        ),
        dims=A._metadata.dims,
        **kwargs
    )
