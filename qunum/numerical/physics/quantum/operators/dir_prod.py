from ..qobjs.torch_qobj import TQobj
from ..qobjs.meta import QobjMeta
from torch import kron

def direct_prod(*args:tuple[TQobj])->TQobj:
    A = args[0]
    if(not isinstance(A, TQobj)):
        A = A[0]
        args = args[0]
        if(not isinstance(A, TQobj)):
            raise TypeError('Must be TQobj')
    
    m = A._metadata.n_particles
    h = A._metadata.hilbert_space_dims
    A = A.detach()
    for i, a in enumerate(args[1:]):
        if(isinstance(a, TQobj)):
            try:
                A = kron(A ,a.detach())
                m+=a._metadata.n_particles
            except:
                ValueError('Must Have Particle Number')
        else:
            raise TypeError('Must be TQobj')
    meta = QobjMeta(n_particles=m, hilbert_space_dims=h, shp=A.shape)
    return TQobj(A, n_particles=m, hilbert_space_dims=h, meta = meta)