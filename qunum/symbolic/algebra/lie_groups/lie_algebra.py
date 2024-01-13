import sympy as sp
import numpy as np
import numba as nb
from typing import Callable


def commutator(op1:sp.Matrix, op2:sp.Matrix)->sp.Matrix:
    return op1 @ op2 - op2 @ op1

def anticommutator(op1:sp.Matrix, op2:sp.Matrix)->sp.Matrix:
    return op1 @ op2 + op2 @ op1

# need to reeval how to do this g_ij is not just a matrix its a tensor of matricies, g_[0,0] = delta[ij] I
def full_apply(ops:sp.Matrix, funct:Callable)->sp.Matrix:
    f = sp.Matrix.zeros(ops[0].shape[0], ops[0].shape[1])
    for i in ops:
        f  += np.sum(list(map(lambda x: funct(i,x), ops)), axis = 0)
    return f

