from torch import jit, Tensor, unique, view_as_complex as toC, view_as_real as toR, concatenate, arange, vmap, einsum, bool as tbool

def addFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis,ix,phi = join_basis(aHat,bHat)
    allW = vmap(lambda x: (x==ix).to(aHat.dtype),out_dims=1)(arange(phi))
    Basis = einsum('AB, Aij->Bij', allW[:aHat.shape[0]], aHat) + einsum('AB, Aij->Bij', allW[aHat.shape[0]:], bHat)
    Coefs = einsum('AB, A -> B', allW[:aHat.shape[0]], aC) + einsum('AB, A -> B', allW[aHat.shape[0]:], bC)
    V = (Basis*Basis.conj()).sum(dim=-1).sqrt().prod(dim=-1)
    Idx = V.to(tbool)
    if(not Idx.any()):
        return 
    else:
        return Basis[Idx]/V[Idx], Coefs[Idx]*V[Idx]

def subFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis,ix,phi = join_basis(aHat,bHat)
    allW = vmap(lambda x: (x==ix).to(aHat.dtype),out_dims=1)(arange(phi))
    Basis = einsum('AB, Aij->Bij', allW[:aHat.shape[0]], aHat) - einsum('AB, Aij->Bij', allW[aHat.shape[0]:], bHat)
    Coefs = einsum('AB, A -> B', allW[:aHat.shape[0]], aC) - einsum('AB, A -> B', allW[aHat.shape[0]:], bC)
    V = (Basis*Basis.conj()).sum(dim=-1).sqrt().prod(dim=-1)
    Idx = V.to(tbool)
    if(not Idx.any()):
        return 
    else:
        return Basis[Idx]/V[Idx], Coefs[Idx]*V[Idx]
     



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