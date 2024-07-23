import numpy as cp 
import numba as nb
import torch

@nb.jit(forceobj = True)
def outter_dp(x:cp.array,X:cp.array,h:cp.array):
    """_summary_

    Args:
        x (cp.array): input_values
        X (cp.array): known_values
        h (cp.array): sigma_parm

    Returns:
        yh (cp.array): (X^{iA} - x^{iB}) \delta{ij}\delta^{A}{}_{C}\delta^{B}{}_{D}  (X^{jC} - x^{jD})
    """    
    return cp.einsum('ABi,i->AB', (cp.einsum('Ai, B->ABi', x, cp.ones(X.shape[0]))-X)**2, h)


@nb.jit(forceobj = True)
def norm_factor(h:cp.array, W:cp.array):
    """_summary_

    Args:
        h (cp.array): h
        W (cp.array): Weights

    Returns:
        nf(cp.array): norm_factor
    """    
    nf = cp.sum(W, axis = 0)
    nf *= cp.prod(cp.sqrt(2*cp.pi)*h)    
    return nf

@nb.jit(forceobj = True)
def gauss(dp:cp.array, W:cp.array, nf:cp.array):
    """_summary_

    Args:
        dp (cp.array): outer dot subtract
        W (cp.array): Weights
        nf (cp.array): norm_factor

    Returns:
        yh(cp.array): _description_
    """  
    return cp.einsum('AB, Bj->Aj', cp.exp(-dp/2), W/nf)
    
@nb.jit(forceobj = True)
def gauss_partition(dp:cp.array, W:cp.array):
    """_summary_

    Args:
        dp (cp.array): _description_
        W (cp.array): _description_
    """
    K = cp.exp(-dp/2)
    B = cp.sum(K, axis = 1)
    return cp.einsum('AB, Bj->Aj', K,W)/B
    #return (cp.dot(K, W).transpose()/B).transpose()
    

@nb.jit(forceobj=True)
def simpsons_integral():
    pass 




@torch.jit.script
def outter_dp_torch(x:torch.Tensor, X:torch.Tensor, h:torch.Tensor)->torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): _description_
        X (torch.Tensor): _description_
        h (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """    
    return torch.einsum('ABi, i -> AB', (torch.einsum('Ai, B->ABi', x, torch.ones_like(X[:,0]))-X)**2, h)

@torch.jit.script
def norm_factor_torch(h:torch.Tensor, W:torch.Tensor)->torch.Tensor:
    """_summary_

    Args:
        h (torch.Tensor): _description_
        W (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """    
    nf = torch.sum(W, dim = 0)
    nf *= torch.prod(torch.sqrt(2*torch.pi)*h)    
    return nf

@torch.jit.script
def gauss_torch(dp:torch.Tensor, W:torch.Tensor, nf:torch.Tensor)->torch.Tensor:
    """_summary_

    Args:
        dp (torch.Tensor): _description_
        W (torch.Tensor): _description_
        nf (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """  
    return torch.einsum('AB, Bi->Ai',torch.exp(-dp/2), W/nf)

@torch.jit.script
def gauss_partition_torch(dp:torch.Tensor, W:torch.Tensor)->torch.Tensor:
    """_summary_

    Args:
        dp (torch.Tensor): _description_
        W (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """    
    K = torch.exp(-dp/2)
    B = torch.sum(K, dim = 1)
    return torch.einsum('AB, Bi-> Ai', K, W)/B
