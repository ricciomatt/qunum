from torch import from_numpy as tensor, ComplexType, dtype as torchDtype, device as torchDevice, arange, zeros, eye, stack, complex128, diag_embed, Tensor, concat, stack
from typing import Generator
from math import sqrt 
from numpy.typing import NDArray

def get_pauli(ret_type:str = 'tqobj', include_identity:bool=True, dtype:torchDtype = complex128, device:torchDevice = 'cpu',**kwargs:dict[str:bool])->Tensor|NDArray:
    if(include_identity):
        sigma = zeros((4,2,2), dtype = dtype)
        sigma[0,0,0] = complex(1,0)
        sigma[0,1,1] = complex(1,0)
        ix = 3
    else:
        sigma = zeros((3,2,2), dtype=dtype)
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
    match kwargs.keys():
        case keys if 'to_tensor' in keys and 'tqobj' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'tqobj' in keys:
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'to_tensor' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
        
    match str(ret_type).lower():
        case 'tqobj':
            from ....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
            return TQobj(sigma, device = device, dtype = dtype)
        case 'tensor':
            return sigma
        case 'numpy':
            return sigma.numpy()
        case _:
            raise TypeError('Only Supports return type of TQobj, Tensor, or Numpy but got {ret}'.format(ret = str(ret_type)))


def get_gellmann(ret_type:str = 'tqobj', include_identity:bool=True, dtype:torchDtype = complex128, device:torchDevice = 'cpu', **kwargs:dict[str:bool])->Tensor|NDArray:
    if(include_identity):
        lam = zeros((9,3,3), dtype= dtype, device = device)
        lam[0] = eye(3, dtype = dtype, device = device)
        ix = 1
    else:
        lam = zeros((8,3,3), dtype=dtype, device= device)
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
    lam[ix]*=1/sqrt(3)

    match kwargs.keys():
        case keys if 'to_tensor' in keys and 'tqobj' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'tqobj' in keys:
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'to_tensor' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
        
    match str(ret_type).lower():
        case 'tqobj':
            from ....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
            return TQobj(lam, device = device, dtype = dtype)
        case 'tensor':
            return lam
        case 'numpy':
            return lam.numpy()
        case _:
            raise TypeError('Only Supports return type of TQobj, Tensor, or Numpy but got {ret}'.format(ret = str(ret_type)))


def su_n_generate(N, ret_type:str = 'generator', oplusu1:bool = True, gen_ret_type:str = 'tensor', dtype:torchDtype = complex128, device:torchDevice = 'cpu', **kwargs:dict[str:bool])->Generator[Tensor|NDArray,None,None]|Tensor|NDArray:
    """
    Computes the generators of the \mathfrak{u}(1) \oplus \mathfrak{su}(N) or \mathfrak{su}(N) Lie algebra.
    
    Parameters:
        N (int): Dimension of the Lie algebra.
        ret_type (str): 'Return Type of the SUN'
    Returns:
        Generator[Tensor|NDArray|TQobj, None,None]|Tensor|TQobj|NDarray: A list of N^2 traceless Hermitian matrices (generators of su(N)).
    """
    def generate_gens(N, include_identity:bool = True, ret_type:str='tensor', dtype:torchDtype=complex128, device:torchDevice = 'cpu'):
        def to_type(v:Tensor, ret_type:str)->Tensor|NDArray:
            match ret_type:
                case 'numpy':
                    return v.numpy()
                case 'tqobj':
                    from ....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
                    return TQobj(v, device = device, dtype = dtype)
                case _:
                    return v
        if include_identity:
            yield eye(N, dtype = dtype, device = device)
        def gen_off_diag(a:int,b:int,N:int)->Generator[Tensor,None,None]:
            M = zeros(N,N, dtype = dtype, device = device)
            M[a-1,b] = 1
            yield M + M.T
            yield -1j*M + 1j*M.T
            
        def gen_diag(a:int, N:int)->Tensor:
            M = zeros((N), dtype = dtype, device=device)
            M[:a] = sqrt(2/(a*(a+1)))
            M[a] = -a*sqrt(2/(a*(a+1)))
            return diag_embed(M)
        for a in range(1,N):
            for b in range(a, N):
                match b:
                    case b if b == a:
                        g = gen_off_diag(a,b,N)
                        for i in g:
                            yield to_type(i, ret_type=ret_type)
                        yield to_type(gen_diag(a,N), ret_type=ret_type)
                    case _:
                        g = gen_off_diag(a,b,N)
                        for i in g:
                            yield to_type(i, ret_type=ret_type)
            
    if('include_identity' in kwargs):
        oplusu1 = kwargs['include_identity']
    generator = (generate_gens(N, include_identity=oplusu1, ret_type=gen_ret_type.lower(), dtype= dtype, device = device))
    match kwargs.keys():
        case keys if 'to_tensor' in keys and 'tqobj' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'tqobj' in keys:
            if(kwargs['tqobj']):
                ret_type = 'tqobj'
        case keys if 'to_tensor' in keys:
            if(kwargs['to_tensor']):
                ret_type='tensor'
    match str(ret_type).lower():
        case 'tqobj':
            from ....physics.quantum.qobjs.dense.core.torch_qobj import TQobj
            return TQobj(stack(list(generator)), device = device, dtype = dtype)
        case 'tensor':
            return stack(list(generator))
        case 'numpy':
            return stack(list(generator)).numpy()
        case 'generator':
            return generator
        case _:
            raise TypeError('Only Supports return type of TQobj, Tensor, Numpy, or Generator[Tensor, None, None] but got {ret}'.format(ret = str(ret_type)))
        
