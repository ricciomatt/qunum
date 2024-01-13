from . import functs
import torch
try:
    import cupy as cp
except:
    import numpy as cp
from typing import Callable
def pdf(x:cp.array, X:cp.array, h:cp.array, W:cp.array)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_

    Returns:
        cp.array: _description_
    """   
    
    x = cp.array(x)
    X = cp.array(X)
    h = cp.array(h)
    W = cp.array(W)
    dp = functs.outter_dp(x,X,h)
    nf = functs.norm_factor(h,W)
    yh = functs.gauss(dp,W,nf)
    return yh

def partition(x:cp.array, X:cp.array, h:cp.array, W:cp.array)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_

    Returns:
        cp.array: _description_
    """
    x = cp.array(x)
    X = cp.array(X)
    h = cp.array(h)
    W = cp.array(W)  
    dp = functs.outter_dp(x,X,h)
    yh = functs.gauss_partition(dp,W)
    return yh

def gauss_torch(x:cp.array, X:cp.array, h:cp.array, W:cp.array)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    x = torch.tensor(x, requires_grad= True,).dtype(torch.cuda.FloatTensor)
    X = torch.tensor(X).dtype(torch.cuda.FloatTensor)
    h = torch.tensor(h).dtype(torch.cuda.FloatTensor)
    W = torch.tensor(W).dtype(torch.cuda.FloatTensor)
    dp = functs.outter_dp_torch(x, X, h)
    nf = functs.norm_factor_torch(h, W)
    yh = functs.gauss_torch(dp,W,nf)
    return cp.array(yh.detach().numpy())

def partiition_torch(x:cp.array, X:cp.array, h:cp.array, W:cp.array)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    x = torch.tensor(x, requires_grad= True,).dtype(torch.cuda.FloatTensor)
    X = torch.tensor(X).dtype(torch.cuda.FloatTensor)
    h = torch.tensor(h).dtype(torch.cuda.FloatTensor)
    W = torch.tensor(W).dtype(torch.cuda.FloatTensor)
    dp = functs.outter_dp_torch(x, X, h)
    yh = functs.gauss_partition_torch(dp, W)
    return cp.array(yh.detach().numpy())



def fisher_torch(x:cp.array, X:cp.array, h:cp.array, W:cp.array, delta:cp.array)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_
        delta (cp.array): _description_

    Returns:
        cp.array: _description_
    """    
    x = torch.tensor(x, requires_grad= True,).dtype(torch.cuda.FloatTensor)
    X = torch.tensor(X).dtype(torch.cuda.FloatTensor)
    h = torch.tensor(h).dtype(torch.cuda.FloatTensor)
    W = torch.tensor(W).dtype(torch.cuda.FloatTensor)
    dp = functs.outter_dp_torch(x, X, h)
    nf = functs.norm_factor_torch(h, W)
    yh = functs.gauss_torch(dp,W,nf)
    return cp.array(yh.detach().numpy())


def diff_entropy(x:cp.array, X:cp.array, h:cp.array, W:cp.array, integrator:Callable)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_
        integrator (Callable): _description_

    Returns:
        cp.array: _description_
    """    
    yh = pdf(x, X, h, W)
    yh = -yh*cp.log(yh)
    x = cp.array(x)
    yh = integrator(yh, x)
    return yh

def kl_divergence(x:cp.array, X:cp.array, h:cp.array, W:cp.array, integrator:Callable)->cp.array:
    """_summary_

    Args:
        x (cp.array): _description_
        X (cp.array): _description_
        h (cp.array): _description_
        W (cp.array): _description_
        integrator (Callable): _description_

    Returns:
        cp.array: _description_
    """    
    yh = pdf(x, X, h, W)
    yh = -yh*cp.log(yh)
    x = cp.array(x)
    yh = integrator(yh, x)
    return 

