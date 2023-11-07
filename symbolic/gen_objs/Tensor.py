import sympy as sp
import numpy as np 
import numba as nb
from numpy.typing import NDArray

class SymTensor:
    def __init__(self, 
                 values:list[list[list[sp.Symbol]]],
                 )->None:
        self.vals = np.array(values)
        self.T = self.vals.T
        self.shape = self.vals.shape
        return
    
    def __add__(self, T:NDArray)->NDArray:
        return self.vals + T
            
    def get_shape(self)->tuple[int]:
        return self.vals.shape
    
    def __matmul__(self, T:NDArray)->NDArray:
        return self.vals @ T

    def adjoint(self):
        return SymTensor(conj(self.T))
    def inv(self):
        return
    
    
        
def conjugate(val):
    return val.conjugate()
conj = np.vectorize(conjugate)

def dot(vals, T):
    return vals @ T