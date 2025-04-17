from torch import einsum, Tensor, unique, view_as_real as toR, jit, arange, vmap, empty_like, tensordot as contract, eye, zeros, stack, dtype as tdp, device as tdevice, tensor, where
from .......mathematics.tensors import levi_cevita_tensor
from .......mathematics.combintorix import EnumerateArgCombos as enumIt, EnumUniqueUnorderedIdx as enumUni
from .......mathematics.algebra.sun import su_n_generate, get_gellmann, get_pauli
from typing import Callable, Iterable
from ..obj.core import LazyTensor
from .core import contractCoef, reduceBasisState, reduceBasisMat
mulCouplings:dict[int:Tensor] = dict()
stateCouplings:dict[tuple[int,str]:Tensor] = dict()


def doMul(aHat:Tensor, aC:Tensor|LazyTensor, bHat:Tensor, bC:Tensor|LazyTensor, N:int,  reduceIt:bool = True,)->tuple[Tensor,Tensor]|tuple[Tensor, LazyTensor]:
    g = getMulCouplings(N, aHat.dtype, aHat.device)
    enum = enumIt(
        range(aHat.shape[0]),
        range(bHat.shape[0])
    )
    idx = enum.__tensor__()
    Basis = einsum('uvb, kAu..., kAv... -> kAb', g, aHat[idx[:,0]], bHat[idx[:,1]])
    
    _, contracted_idx = unique(
            toR(
                Basis
            ).any(
                dim=tuple(
                    range(
                        2,
                        len(
                            Basis.shape
                        )
                    )
                )
            ),
            return_inverse=True, 
            dim=0
        )

    uidx = unique(
        contracted_idx
    )
    V:Tensor = vmap(lambda x: (
        x==contracted_idx
    ).to(aHat.dtype),  out_dims=1)(uidx)
    Basis = einsum('AB, A...-> B...',V/V.sum(0), Basis)
    Coef = contractCoef(aC, bC, idx, enum, V)
    if(reduceIt):
        return reduceBasisMat(Basis, Coef)
    else:
        return Basis, Coef
    

    
def doMulStateMat(aHat:Tensor, aC:Tensor, psiHat:Tensor, psiC:Tensor, psiTp:str, N:int, reduceIt:bool = True, renorm:bool = False) -> tuple[Tensor, Tensor]|tuple[Tensor,LazyTensor]:
    g = getStateOperCouplings(N, aHat.dtype, aHat.device)
    if(psiTp == 'bra'):
        g = g.conj()
    enum = enumIt(
        range(aHat.shape[0]),
        range(psiHat.shape[0])
    )
    idx = enum.__tensor__()
    Basis = einsum('uab, kAu..., kAa...-> kAb...', g, aHat[idx[:,0]], psiHat[idx[:,1]])
    
    _, contracted_idx = unique(
            toR(
                Basis
            ).any(
                dim=-1
            ),
            return_inverse=True, 
            dim=0
        )
    
    uidx = unique(contracted_idx)
    V:Tensor = vmap(lambda x: (x==contracted_idx).to(psiC.dtype), out_dims = 1)(uidx) 
    Basis = einsum('AB, B...->A...',V, Basis)
    Coefs = contractCoef(aC, psiC, idx, enum, V,)
    if(reduceIt):
        return reduceBasisState(Basis, Coefs, renorm=renorm)
    else:
        return Basis, Coefs

def doMulStateState(betaHat:Tensor, betaC:Tensor, betaTp:str, alphaHat:Tensor, alphaC:Tensor, alphaTp:str, N:int)->Tensor:
    match (betaTp, alphaTp):
        case ('ket','bra'):
            idx = enumIt(range(betaHat.shape[0]), range(betaHat.shape[0])).__tensor__()
            Cfs = betaC[idx[:,1]] * alphaC[idx[:,0]]
            return contractCoef(
                Cfs, 
                (
                    alphaHat[idx[:,0]]*betaHat[idx[:,1]]
                ).sum(dim=-1).ceil().all(dim=-1).to(alphaC.dtype)
                
            ).sum(
                dim=0
            )
        case _:
            raise TypeError('State must be \\bra{\\beta}\\ket{\\alpha} or \\ket{\\beta}\\bra{\\alpha}')

@jit.script
def mul_state_mat(psiHat:Tensor, aHat:Tensor, aC:Tensor)->Tensor:
    retHat = einsum('Ai, BAj, B-> Aij', (psiHat[:,0]), aHat[:,:,0], aC)
    m = empty_like(psiHat)
    m[:,0] = retHat[:,0,0] + retHat[:, 0,3] + retHat[:, 1,1] - 1j* retHat[:,1,2]
    m[:,1] = retHat[:,1,1] - retHat[:, 1,3] + retHat[:, 0,1] + 1j* retHat[:,0,2]
    return m

@jit.script
def expectation_value(betaHat:Tensor, alphaHat:Tensor)->Tensor:
    return (betaHat*alphaHat).sum()

def getMulCouplings(N:int, dtype:tdp, device:tdevice)->Tensor:
    if(N in mulCouplings):
        return mulCouplings[N].to(dtype=dtype, device = device)
    else:
        for i in mulCouplings:
            del mulCouplings[i]
        g = su_n_generate(N, ret_type='tensor', include_identity= True, dtype=dtype, device = device,)
        mulCouplings[N] = einsum('Aij, Bjk, Cki->ABC', g, g, g)/N
        return mulCouplings[N].to(dtype=dtype, device = device)

def getStateOperCouplings(N:int, dtype:tdp, device:tdevice)->Tensor:
    if(N in stateCouplings):
        return stateCouplings[N].to(dtype = dtype, device= device)
    else:
        for i in stateCouplings:
            del stateCouplings[i]
        g = su_n_generate(N, include_identity= True, dtype=dtype, device = device, to_tensor=True, tqobj=False)
        stateCouplings[N] = g
        return stateCouplings[N]
