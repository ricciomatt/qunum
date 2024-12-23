import numpy as np
"""try:
    import cupy as cp 
except:
    import numpy as cp
"""
import numpy as cp
import numba as nb
import torch 

@torch.jit.script
def torch_simpsons_1d_1_3(dx:float, num:int= 3):
    S = torch.ones(num, dtype=torch.complex64) * dx/3
    for i in range(1,num-1):
        if((i%2) == 0):
            S[i]*=2
        else:
            S[i]*=4
    return S

@nb.njit(parallel = True, fastmath = True)
def integral_simpson_vect_1_3(yh:np.array, xin:np.array, num_pts:int, delta:np.array)->np.array:
    """_summary_

    Args:
        yh (np.array): _description_
        xin (np.array): _description_
        num_pts (int): _description_
        delta (np.array): _description_

    Returns:
        np.array: _description_
    """    
    S = np.ones((yh.shape[0]), dtype = np.float64)
    for j in range(xin.shape[1]):
        for A in nb.prange(yh.shape[0]):
            if((int(np.floor(A/(num_pts**j))) == 0 )or (int(np.floor(A/(num_pts**j))) == yh.shape[0]-1)):
                S[A] *= 1*(delta[j]/3)
            elif(int(np.floor(A/(num_pts**j)))%2 == 0):
                S[A] *= 2*(delta[j]/3)
            else:
                S[A] *= 4*(delta[j]/3)
    return S


@nb.njit(parallel = True, fastmath = True)
def integral_simpson_vect_3_8(yh:np.array, xin:np.array, num_pts:int, delta:np.array)->np.array:
    """_summary_

    Args:
        yh (np.array): _description_
        xin (np.array): _description_
        num_pts (int): _description_
        delta (np.array): _description_

    Returns:
        np.array: _description_
    """    
    S = np.ones((yh.shape[0]), dtype = np.float64)
    for j in range(xin.shape[1]):
        for A in nb.prange(yh.shape[0]):
            if((int(np.floor(A/(num_pts**j))) == 0 )or (int(np.floor(A/(num_pts**j))) == yh.shape[0]-1)):
                S[A] *= 1*(delta[j]/3)
            elif(int(np.floor(A/(num_pts**j)))%2 == 0):
                S[A] *= 2*(3*delta[j]/8)
            else:
                S[A] *= 3*(3*delta[j]/8)
    return S




@nb.jit(forceobj=True)
def simpsons_1_3(f:cp.array, xin:cp.array)->cp.array:
    """_summary_

    Args:
        f (cp.array): _description_
        xin (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    h = (xin[xin.shape[0]-1]-xin[0])/f.shape[0]
    N = xin.shape[0]//xin.shape[1]
    S = cp.array(integral_simpson_vect_1_3(f.get(), xin.get(), N,  h.get()))
    return S @ f

@nb.jit(forceobj=True)
def simpsons_3_8(f:cp.array, xin:cp.array)->cp.array:
    """_summary_

    Args:
        f (cp.array): _description_
        xin (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    print('running')
    h = (xin[xin.shape[0]-1]-xin[0])/f.shape[0]
    N = xin.shape[0]//xin.shape[1]
    S = cp.array(integral_simpson_vect_3_8(f.get(), xin.get(), N,  h.get()))
    return S @ f  
    
def riemann_sum(f:cp.array, xin:cp.array)->cp.array:
    return f.sum(axis=0)*cp.prod((xin[1]-xin[0]))
    

def monte_carlo(f:cp.array, xin:cp.array)->cp.array:
    """_summary_

    Args:
        f (cp.array): _description_
        a (cp.array): _description_
        b (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    
    return cp.prod((xin.max(axis = 0)-xin.min(axis = 0))/xin.shape[0])*f.sum(axis = 0)
    