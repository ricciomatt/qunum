from ...qobjs.opers import Operator
from sympy import Matrix, kronecker_product as d_prod, Symbol, sin, I, cos
from .....numerical.algebra.representations.su import get_pauli
def rn(theta:Symbol|Matrix, n_vec:Matrix, n_particles:int):
    sigma = [Operator(i) for i  in get_pauli(include_identity=True)]
    A = Matrix.zeros(2,2)
    try:
        theta[0]
        for i in range(1,4):
            A-=n_vec[i-1]*sigma[i]*I*sin(theta[i])
    except:
        for i in range(1,4):
            A-=n_vec[i-1]*sigma[i]*I*sin(theta)
    A+= sigma[0] * cos(theta)
    return Operator(A, n_particles = 1, hilbert_space_dims = 2)

def r_dir(theta:float|Symbol, dir = 'x'):
    sigma = [Operator(i) for i  in get_pauli(include_identity=True)]
    if(dir == 'x'):
        dir = 1
    elif(dir == 'y'):
        dir = 2
    elif(dir == 'z'):
        dir = 3
    else:
        raise ValueError('must be x,y,or z')
    return Operator(sigma[0] * cos(theta) -I*sigma[dir]*sin(theta), n_particles = 1, hilbert_space_dims = 2)