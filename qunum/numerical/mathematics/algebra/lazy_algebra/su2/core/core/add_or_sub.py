from torch import jit, Tensor, unique, view_as_complex as toC, view_as_real as toR, concatenate, arange, vmap, einsum

def addFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis, Ix = join_basis(aHat=aHat, bHat=bHat)
    aW = vmap(lambda x: (x==Ix[:aC.shape[0]]).to(aC.dtype), out_dims=1)(arange(Basis.shape[0]))
    bW = vmap(lambda x: (x==Ix[aC.shape[0]:]).to(aC.dtype), out_dims=1)(arange(Basis.shape[0]))
    return Basis, einsum('Ai, A ->i ', aW, aC) + einsum('Ai, A ->i ', bW, bC)

def subFun(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis, Ix = join_basis(aHat=aHat, bHat=bHat)
    aW = vmap(lambda x: (x==Ix[:aC.shape[0]]).to(aC.dtype), out_dims=1)(arange(Basis.shape[0]))
    bW = vmap(lambda x: (x==Ix[aC.shape[0]:]).to(aC.dtype), out_dims=1)(arange(Basis.shape[0]))
    return Basis, einsum('Ai, A ->i ', aW, aC) - einsum('Ai, A ->i ', bW, bC)

@jit.script
def join_basis(aHat:Tensor,  bHat:Tensor)->tuple[Tensor,Tensor]:
    jnd = unique(
        toR(
            concatenate(
                (
                    aHat, 
                    bHat
                ), 
                dim=0
            )
        ), 
        dim=0, 
        return_inverse=True
    )
    return toC(jnd[0]), jnd[1]