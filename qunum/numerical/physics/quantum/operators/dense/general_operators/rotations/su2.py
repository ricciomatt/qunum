from .....qobjs.dense.core.torch_qobj import TQobj, direct_prod
from .......mathematics.algebra.sun import get_pauli
from numpy import cos, sin
def rn(theta, n_vec, n_particles:int)->TQobj:
    sigma = [TQobj(i, n_particles =1, hilbert_space_dims = 2) for i  in get_pauli(include_identity=True)]
    pass
def r_dir(theta:float, dir:str = 'x')->TQobj:
    sigma = [TQobj(i, n_particles =1, hilbert_space_dims = 2) for i  in get_pauli(include_identity=True)]
    ix = {'x':1, 'y':2, 'z':3}
    if(dir not in ix):
        raise ValueError('must be x,y,or z')
    return TQobj(sigma[0] * cos(theta) - sigma[ix[dir]]*sin(theta)*complex(0,1), n_particles = 1, hilbert_space_dims = 2)

def r_euler(alpha:float=0., beta:float=0., gamma:float = 0.)->TQobj:
    return r_dir(alpha, 'z') @ r_dir(beta,'y') @ r_dir(gamma, 'z')


def decompose(A:TQobj)->TQobj:
    return