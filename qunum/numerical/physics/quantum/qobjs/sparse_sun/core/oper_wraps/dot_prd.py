from ..core import doMul, doMulStateMat, doMulStateState
from ..obj import SUNMatrix, SUNState
from torch import Tensor
from .......mathematics.tensors.lazy import LazyTensor
def SparseDot(a:SUNMatrix|SUNState, b:SUNMatrix|SUNState)->SUNMatrix|SUNState|Tensor|LazyTensor:
    assert (isinstance(a, SUNMatrix) or isinstance(a,SUNState)) and (isinstance(b, SUNMatrix) or isinstance(b,SUNState)), TypeError('Sparse Dot is only for SparseSUN Objects')
    assert (a.N == b.N), Exception('Sparse Dot is only for SparseSUN Objects with matching N found a.N={N1} and b.N = {N2}'.format(N1 = str(a.N), N2 = str(b.N)))
    match (a, b):
        case (SUNMatrix(), SUNMatrix()):
            return SUNMatrix(*doMul(a.basis, a.coefs, b.basis, b.coefs, N = a.N), N= a.N)
        case (SUNMatrix(), SUNState()):
            assert b.objTp == 'ket', Exception('dot(a,psi), only supported if psi is ket type')
            return SUNState(*doMulStateMat(a.basis, a.coefs, b.basis, b.coefs, b.objTp, N=a.N), N=b.N, objTp=b.objTp)
        case (SUNState(), SUNMatrix()):
            assert a.objTp == 'bra', Exception('dot(a,psi), only supported if psi is bra type')
            return SUNState(*doMulStateMat(a.basis, a.coefs, b.basis, b.coefs), N = b.N, objTp=a.objTp)
        case (SUNState(), SUNState()):
            return doMulStateState(a.basis, a.coefs, b.basis, b.coefs)
        case _:
            raise Exception('Sparse Dot is only for SparseSUN Objects'.format(N1 = str(a.N), N2 = str(b.N)))