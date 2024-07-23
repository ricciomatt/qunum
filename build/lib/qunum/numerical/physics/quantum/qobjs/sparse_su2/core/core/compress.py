from torch import Tensor, unique, view_as_real as toR, view_as_complex as toC, vmap, arange, einsum
def reduce(aHat:Tensor, aC:Tensor)->tuple[Tensor,Tensor]:
    A, Ix = unique(
        toR(aHat).any(dim=-1), 
        dim=0, 
        return_inverse = True
    )
    Ix = vmap(lambda x: (Ix==x).to(aHat.dtype), in_dims=1)(arange(A.shape[0]))
    return einsum('AB, Aij -> Bij', Ix, aHat), einsum('AB, A -> B', Ix, aC)

