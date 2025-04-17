from ..core import doMul
from ..obj import SU2Matrix
def dot(a:SU2Matrix, b:SU2Matrix)->SU2Matrix:
    return SU2Matrix(*doMul(a.basis, a.coefs, b.basis, b.coefs))