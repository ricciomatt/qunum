from ..core.torch_qobj import TQobj
from typing import Iterable
from .zeros_c import zeros_from_context


def mk_basis(state:int, n_particles:int = 1, hilbert_space_dims:int = 2, shp:int|Iterable[int]|None = None, dims:dict[int:int]|None = None, obj_tp:str = 'ket', **kwargs)->TQobj:
    assert obj_tp != 'scaler', 'Must be bra ket or operator'
    match obj_tp:
        case 'operator':
            t = 'ket'
        case _:
            t = obj_tp
    psi = zeros_from_context(shp=shp, n_particles=n_particles, hilbert_space_dims=hilbert_space_dims, dims=dims, obj_tp=t, **kwargs)
    if(state >= psi.shape[0]):
        raise IndexError(
            '''Error {index} out of range for {shp}. State must be less than n_particles**hilbert_space_dims = {l}. Object_tp='''.format(
                index = state, 
                shp=str(psi.shape), 
                l = int(n_particles**hilbert_space_dims),
                obj_tp=obj_tp
            )
    )
    match obj_tp:
        case 'ket':
            psi[..., state, 0] = 1
        case 'bra':
            psi[..., 0, state] = 1
        case 'operator':
            psi[..., state, 0] = 1
            psi = psi.to_density()
    return psi
    
