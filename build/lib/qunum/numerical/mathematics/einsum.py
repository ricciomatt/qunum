from torch import einsum as tein, jit, Tensor
from ..physics.quantum.qobjs import TQobj
from ..physics.quantum.qobjs.meta.meta import QobjMeta
from typing import Tuple

def einsum(indicies:str, *args:Tuple[Tensor|TQobj,...], **kwargs)->TQobj:
    ret_ten = tein(indicies, *args)
    meta = None
    for a in args:
        if(isinstance(a, TQobj)):
            meta = QobjMeta(n_particles=a._metadata.n_particles, dims=a._metadata.dims, shp=ret_ten.shape)
            break
    assert meta is not None, 'If you are looking to just do a regular einstien summation use torch.einsum(*args) not qn.einsum'
    if(isinstance(ret_ten, TQobj)):
        ret_ten.set_meta(meta)    
    else:
        ret_ten = TQobj(ret_ten, meta=meta)
    return ret_ten