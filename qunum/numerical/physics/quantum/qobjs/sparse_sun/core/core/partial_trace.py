from torch import Tensor, tensor as createTensor
from .core import contractCoef

def ptrace(aHat:Tensor, aC:Tensor, keep_ix:Tensor, TrOut:Tensor)->Tensor:  
    Cfs = contractCoef(aC[TrOut], aHat[:, TrOut, 0].prod(dim = 1), dims =([0],[0]))
    return aHat[:, keep_ix], ((aHat.shape[-1]**(TrOut.shape[0]/2)))*Cfs
    
    
def fullTrace(aHat:Tensor, aC:Tensor)->Tensor:
    return contractCoef(aC, aHat[..., 0].prod(dim = 1), dims =([0],[0]))

