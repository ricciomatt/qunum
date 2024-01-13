from sympy import kronecker_product as dprod
from ...qobjs.sqobj import SQobj

def direct_prod(A:SQobj, B:SQobj):
    if(not isinstance(A,SQobj) or not isinstance(B,SQobj)):
        raise TypeError('Must Be SQobj type')
    if(A._metadata.hilbert_space_dims == B._metadata.hilbert_space_dims):
        return SQobj(dprod(A,B), n_particles=A._metadata.n_particles+B._metadata.n_particles, 
                     hilbert_space_dims=A._metadata.hilbert_space_dims)
    else:
        raise ValueError('Hilbert Space Dimensions must match')