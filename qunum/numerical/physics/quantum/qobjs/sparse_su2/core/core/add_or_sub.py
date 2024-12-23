from torch import jit, Tensor, unique, view_as_real as toR, concatenate, arange, vmap, einsum
from .compress import reduce

def subFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    return addFun(aHat, aC, -1*bHat, -1*bC)

def addFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis,ix,phi = join_basis(aHat,bHat)
    allW = vmap(lambda x: (x==ix).to(aHat.dtype),out_dims=1)(arange(phi))
    Basis = einsum('AB, Aij->Bij', allW[:aHat.shape[0]], aHat) + einsum('AB, Aij->Bij', allW[aHat.shape[0]:], bHat)
    Coefs = einsum('AB, A -> B', allW[:aHat.shape[0]], aC) + einsum('AB, A -> B', allW[aHat.shape[0]:], bC)
    return  reduce(Basis, Coefs)

@jit.script
def join_basis(aHat:Tensor,  bHat:Tensor)->tuple[Tensor,Tensor, int]:
    jnd = concatenate(
                (
                    aHat, 
                    bHat
                ), 
                dim=0
            )
    
    K, ix = unique(
        toR(
            jnd
        ).any(dim=-1), 
        dim=0, 
        return_inverse=True
    )
    return jnd, ix, K.shape[0]

