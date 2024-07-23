from ..core import doMul
from ..obj import PauliMatrix
def dot(a:PauliMatrix, b:PauliMatrix)->PauliMatrix:
    return PauliMatrix(*doMul(a.basis, a.coefs, b.basis, b.coefs))