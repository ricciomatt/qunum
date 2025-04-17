from torch import  Tensor, unique, view_as_real as toR, jit, arange, vmap, empty_like, tensordot as contract, eye, zeros, stack, dtype as tdp, device as tdevice, tensor, where, einsum, bool as tbool
from typing import Iterable
from ..obj.core import LazyTensor, contractLazy, einsumLazy
from .......mathematics.combintorix import EnumerateArgCombos


def contractCoef(a:Tensor|LazyTensor, b:Tensor|LazyTensor, idx:Tensor, enum:EnumerateArgCombos, Projection:Tensor, **kwargs)->Tensor|LazyTensor:
    match (a,b):
        case (a,b) if isinstance(a, LazyTensor) and isinstance(b, LazyTensor):
            a = a[enum.set_dim(dim = 0, inplace=False)]*b[enum.set_dim(dim = 1, inplace=False)]
        case (a,b) if isinstance(a, LazyTensor) and isinstance(b, Tensor):
            a =  a[enum.set_dim(dim = 0, inplace=False)]*b[idx[:,1]]
        case (a,b) if isinstance(a, Tensor) and isinstance(b, LazyTensor):
            a =  a[idx[:,0]]*b[enum.set_dim(dim = 1, inplace=False)]
        case (a,b) if isinstance(a, Tensor) and isinstance(b, Tensor):
            a = a[idx[:,0]]*b[idx[:,1]]
    return einsumLazy('A..., AB->B...', a, Projection, warnMe = False)

def reduceBasisState(Basis:Tensor, Coefs:Tensor|LazyTensor, renorm:bool = False)->tuple[Tensor, Tensor|LazyTensor]:
    V = (Basis*Basis.conj()).sum(dim=-1)
    Idx = toR(V).abs().sum(-1).to(tbool).all(-1)
    if(not Idx.any()):
        return None, None
    else:
        Basis = Basis[Idx]
        Coefs = Coefs[Idx]
        V = (Basis*Basis.conj()).sum(-1).prod(-1).pow(1/(2*Basis.shape[1]))
        match (Coefs, renorm):
            case (LazyTensor(), False):
                Coefs = einsumLazy('A..., A -> A...', Coefs, V)
            case (LazyTensor(), True):
                Coefs:LazyTensor = einsumLazy('A..., A -> A...', Coefs, V)
                Coefs = einsumLazy('A..., ...->A...', Coefs, 1/(Coefs.abssqr().sum(dim=0).sqrt()))
            case (Tensor(), True):
                Coefs = einsum('A..., A -> A...', Coefs, V)
                Coefs = einsum('A..., ...->A...', Coefs, 1/(Coefs*Coefs.conj().sum(dim=0).sqrt()))
            case (Tensor(), False):
                Coefs = einsum('A..., A -> A...', Coefs, V)
            case _:
                TypeError('Type of Coef not recognized, type(Coef) = {tp}, but must be LazyTensor or torch.Tensor'.format(tp=type(Coefs)))
        return einsum('A..., A -> A...', Basis, 1/V), Coefs

def reduceBasisMat(Basis:Tensor, Coefs:Tensor|LazyTensor)->tuple[Tensor, Tensor|LazyTensor]:
    Idx = toR((Basis*Basis.conj()).sum(dim=-1)).abs().sum(-1).to(tbool).all(-1)
    if(not Idx.any()):
        return None, None
    else:
        Basis = Basis[Idx]
        V = (Basis*Basis.conj()).real.sqrt().max(dim=2)[0]
        return einsum('ABc..., AB...->ABc...', Basis, 1/V ), einsumLazy('A..., A...->A...',Coefs[Idx], V.prod(1))