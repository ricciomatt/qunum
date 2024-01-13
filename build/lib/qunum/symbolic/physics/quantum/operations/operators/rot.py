from ...qobjs.sqobj import SQobj
from sympy import Matrix, kronecker_product as d_prod, Symbol, sin, I, cos
from ......numerical.algebra.representations.su import get_pauli
def rn(theta:Symbol|Matrix, n_vec:Matrix, n_particles:int)->SQobj:
    sigma = [SQobj(i, n_particles =1, hilbert_space_dims = 2) for i  in get_pauli(include_identity=True)]
    A = Matrix.zeros(2,2)
    try:
        theta[0]
        for i in range(1,4):
            A-=n_vec[i-1]*sigma[i]*I*sin(theta[i])
    except:
        for i in range(1,4):
            A-=n_vec[i-1]*sigma[i]*I*sin(theta)
    A+= sigma[0] * cos(theta)
    print(A)
    return SQobj(A, n_particles = 1, hilbert_space_dims = 2)

def r_dir(theta:float|Symbol, dir = 'x')->SQobj:
    sigma = [SQobj(i, n_particles =1, hilbert_space_dims = 2) for i  in get_pauli(include_identity=True)]
    ix = {'x':1, 'y':2, 'z':3}
    if(dir not in ix):
        raise ValueError('must be x,y,or z')
    return SQobj(sigma[0] * cos(theta) -sigma[ix[dir]]*sin(theta)*I, n_particles = 1, hilbert_space_dims = 2)

def r_euler(alpha:None|Symbol=0, beta:None|Symbol=0, gamma:None|Symbol = None)->SQobj:
    return r_dir(alpha, 'z') @ r_dir(beta,'y') @ r_dir(gamma, 'z')