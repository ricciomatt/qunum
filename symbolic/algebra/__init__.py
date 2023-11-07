import sympy as sp
from sympy import I, E
import numpy as np 
from .lie_groups import *
import numpy as np
from numpy.typing import NDArray
def get_pauli(include_identity:bool=True)->NDArray[np.complex64]:
    if(include_identity):
        sigma = np.zeros((4,2,2), dtype = np.complex64)
        sigma[0,0,0] = complex(1,0)
        sigma[0,1,1] = complex(1,0)
        ix = 3
    else:
        sigma = np.zeros((3,2,2), dtype=np.complex64)
        ix = 2
    sigma[ix,0,0] = complex(1,0)
    sigma[ix,1,1] = complex(-1,0)
    ix-=1
    sigma[ix,0,1] = complex(0,-1)
    sigma[ix,1,0] = complex(0,1)
    ix-=1
    sigma[ix,0,1] = complex(1,0)
    sigma[ix,1,0] = complex(1,0)
    ix-=1
    return sigma

def get_gellmann(include_identity:bool= True)->NDArray[np.complex64]:
    if(include_identity):
        lam = np.zeros((9,3,3), dtype=np.complex64)
        for i in range(3):
            lam[0,i,i] = complex(1,0)
        
        ix = 1
    else:
        lam = np.zeros((8,3,3), dtype=np.complex64)
        ix = 0
    
    
    lam[ix,0,1] = complex(1,0)
    lam[ix,1,0] = complex(1,0)
    ix+=1
    lam[ix,0,1] = complex(0,-1)
    lam[ix,1,0] = complex(0,1)
    ix+=1

    lam[ix,0,0] = complex(1,0)
    lam[ix,1,1] = complex(-1,0)
    ix+=1

    lam[ix,0,2] = complex(1,0)
    lam[ix,2,0] = complex(1,0)
    ix+=1

    lam[ix,0,2] = complex(0,-1)
    lam[ix,2,0] = complex(0,1)
    ix+=1

    lam[ix,1,2] = complex(1,0)
    lam[ix,2,1] = complex(1,0)
    ix+=1

    lam[ix,1,2] = complex(0,-1)
    lam[ix,2,1] = complex(0,1)
    ix+=1

    lam[ix,0,0] = complex(1,0)
    lam[ix,1,2] = complex(1,0)
    lam[ix,2,1] = complex(-2,0)
    lam[ix]*=1/np.sqrt(3)

    return lam

gell_man = [
    sp.Matrix(((1,0,0),
                   (0,1,0),
                   (0,0,1))),
    sp.Matrix(((0,1,0),
               (1,0,0),
               (0,0,0))), 
      sp.Matrix(
          (
              (0,-I, 0),
              (I, 0, 0),
              (0,0,0),
      )),
      sp.Matrix(
          (
              (1,0,0),
              (0,-1,0),
              (0,0,0),
          )
      ),
       sp.Matrix(
           (
               (0,0,1),
               (0,0,0),
               (1,0,0)
            )
        ), 
      sp.Matrix(
          (
              (0, 0, -I),
              (0, 0, 0),
              (I, 0 ,0),
      )),
      sp.Matrix(
          (
              (0,0,0),
              (0,0,1),
              (0,1,0),
          )
      ),
      sp.Matrix(
          (
              (0,0,0),
              (0,0,-I),
              (0,I,0),
          )
      ),
      1/(sp.sqrt(3))*sp.Matrix(
          (
              (1,0,0),
              (0,1,0),
              (0,0,-2),
          )
      )
    ]



pauli = [
sp.Matrix(
    [
        [1,0],
        [0,1]
     ]
),
sp.Matrix([
    [0, 1],
    [1, 0]
]),
sp.Matrix([
    [0, -I],
    [I, 0]
]),
sp.Matrix([
    [1, 0],
    [0, -1]
])]