from torch import Tensor, view_as_real as toR, einsum, bool as tbool, zeros
def reduce(Basis:Tensor, Coefs:Tensor)->tuple[Tensor,Tensor]:
    Idx = toR((Basis*Basis.conj()).sum(dim=-1)).abs().sum(-1).to(tbool).all(-1)
    if(not Idx.any()):
        return 
    elif(not Idx.all()):
        V = Basis[Idx].abs().max(-1)[0].max(-1)[0]
        return einsum('Aij, A -> Aij', Basis[Idx], V.pow(-1)), einsum('A, A->A', Coefs[Idx], V)
    else:
        V = Basis[Idx].abs().max(-1)[0].max(-1)[0]
        return einsum('Aij, A -> Aij', Basis[Idx], V.pow(-1)), einsum('A, A->A', Coefs[Idx], V)

def reduce_state(Basis:Tensor, Coefs:Tensor)->tuple[Tensor,Tensor]:
    V = (Basis*Basis.conj()).sum(dim=-1)
    Idx = toR(V).abs().sum(-1).to(tbool).all(-1)
    if(not Idx.all()):
        return 
    else:
        V = (V[Idx].prod(dim=-1))
        return einsum('Aij, A -> Aij', Basis[Idx], (V).pow(-1)), einsum('A, A->A', Coefs[Idx], V)
