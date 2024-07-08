import numpy as np
from torch import from_numpy as tensor, ComplexType
from numpy.typing import NDArray

def get_pauli(to_tensor:bool= False, include_identity:bool=True, dtype = np.complex128, tqobj:bool=True)->NDArray[np.complex64|np.complex128]|ComplexType:
    if(include_identity):
        sigma = np.zeros((4,2,2), dtype = dtype)
        sigma[0,0,0] = complex(1,0)
        sigma[0,1,1] = complex(1,0)
        ix = 3
    else:
        sigma = np.zeros((3,2,2), dtype=dtype)
        ix = 2
    sigma[ix,0,0] = complex(1,0)
    sigma[ix,1,1] = complex(-1,0)
    ix-=1
    sigma[ix,1,0] = complex(0,1)
    sigma[ix,0,1] = complex(0,-1)
    ix-=1
    sigma[ix,0,1] = complex(1,0)
    sigma[ix,1,0] = complex(1,0)
    ix-=1
    if(tqobj):
        from .....physics.quantum.qobjs.torch_qobj import TQobj
        return TQobj(sigma)
    elif(to_tensor):
        return tensor(sigma)
    else:
        return sigma
    

def get_gellmann(to_tensor:bool = False, include_identity:bool= True, tqobj:bool = False)->NDArray[np.complex64]|ComplexType:
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

    if(tqobj):
        from .....physics.quantum.qobjs.torch_qobj import TQobj
        return TQobj(lam)
    elif(to_tensor):
        return tensor(lam)
    else:
        return lam

