import numpy as np 
try:
    import cupy as cp 
    from cupy.typing import NDArray
except:
    import numpy as np
    from numpy.typing import NDArray
from typing import Callable

def lin_reg_do(X:NDArray, Y:NDArray, 
               function:Callable, 
               inverse_function:Callable):
    XtXI = np.linalg.inv(X.T@ X)
    beta = XtXI @ (X.T @ inverse_function(Y))
    return beta
