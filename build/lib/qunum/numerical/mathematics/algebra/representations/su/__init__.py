import numpy as np
from torch import from_numpy as tensor, ComplexType, dtype as torchDtype, device as torchDevice, arange, zeros, eye, stack, complex128, diag_embed, Tensor
from numpy.typing import NDArray
from typing import Generator

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
        from .....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
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
    lam[ix,1,1] = complex(1,0)
    lam[ix,2,2] = complex(-2,0)
    lam[ix]*=1/np.sqrt(3)

    if(tqobj):
        from .....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
        return TQobj(lam)
    elif(to_tensor):
        return tensor(lam)
    else:
        return lam
    
def su_n_generate(N, include_identity:bool = True, dtype:torchDtype = complex128, device:torchDevice = 'cpu', to_tensor:bool = True, tqobj:bool = False)->Generator[Tensor,None,None]|Tensor:
    """
    Computes the generators of the su(N) Lie algebra.
    
    Parameters:
        N (int): Dimension of the Lie algebra.
    
    Returns:
        list: A list of N^2 traceless Hermitian matrices (generators of su(N)).
    """
    def generate_gens(N, include_identity:bool = True, dtype:torchDtype=complex128, device:torchDevice = 'cpu'):
        if include_identity:
            yield eye(N, dtype = dtype, device = device)
        for i in range(N):
            for j in range(i + 1, N):
                # Real symmetric part
                M = zeros((N, N), dtype=dtype, device = device)
                M[i, j] = 1
                yield M + M.T
                yield -1j*M  + 1j*M.T
        # Step 2: Diagonal (traceless) generators
        for k in range(1, N):
            M = zeros((N), dtype=dtype, device = device)
            M[:k] = 1
            M[k] = -k
            yield diag_embed(M/((M*M/2).sum().sqrt()))
            
            
    generator = (generate_gens(N, include_identity=include_identity, dtype= dtype, device = device))
    
    if(to_tensor):
        generator = list(generator)
        if(N == 3):
            e = generator.pop(8)
            generator.insert(3,e)
        return stack(generator)
    elif(tqobj):
        from .....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
        return TQobj(stack(generator))
    else:
        return generator

