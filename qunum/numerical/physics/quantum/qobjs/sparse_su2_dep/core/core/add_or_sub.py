from torch import jit, Tensor, unique, view_as_real as toR, concatenate, arange, vmap, einsum
from typing import Callable
from ..obj.core import LazyTensor
from .core import contractCoef, reduceBasisState, reduceBasisMat
def subFun(aHat:Tensor, aC:Tensor|LazyTensor, bHat:Tensor, bC:Tensor|LazyTensor, functional:bool = False)->tuple[Tensor,Tensor]:
    return addFun(aHat, aC, -1*bHat, -1*bC)

def addFun(aHat:Tensor, aC:Tensor|LazyTensor, bHat:Tensor, bC:Tensor|LazyTensor, functional:bool = False)->tuple[Tensor,Tensor]:
    assert aHat.shape[-1] == bHat.shape[-1], TypeError('Only Support Matrix Matrix and State State addition and subtraction')
    Basis, ix, phi = join_basis(aHat,bHat)
    allW = vmap(
        lambda x: 
        (x==ix).to(aHat.dtype),
            out_dims=1
    )(
        arange(phi)
    )
    Basis = einsum('AB, Aij-> Bij', allW[:aHat.shape[0]], aHat) + einsum('AB, Aij-> Bij', allW[aHat.shape[0]:], bHat)
    Coefs = contractCoef(aC, allW[:aHat.shape[0]], dims = ([0],[0])) + contractCoef(bC, allW[aHat.shape[0]:], dims = ([0],[0]))
    match Basis.shape[-1]:
        case 2:
            return reduceBasisState(Basis, Coefs)
        case 4:
            return reduceBasisMat(Basis, Coefs)
        case _:
            raise TypeError('Only Support Matrix Matrix and State State addition and subtraction')

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

