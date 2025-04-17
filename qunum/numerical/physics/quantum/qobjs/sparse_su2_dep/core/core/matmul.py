from torch import einsum, Tensor, unique, view_as_real as toR, jit, arange, vmap, empty_like, tensordot as contract, eye, zeros, stack, dtype as tdp, device as tdevice, tensor, where
from .......mathematics.tensors import levi_cevita_tensor
from .......mathematics.combintorix import EnumerateArgCombos as enumIt, EnumUniqueUnorderedIdx as enumUni
from typing import Callable, Iterable
from ..obj.core import LazyTensor
from .core import contractCoef, reduce, reduce_state, reduceBasisState, reduceBasisMat

def doMul(aHat:Tensor, aC:Tensor|LazyTensor, bHat:Tensor, bC:Tensor|LazyTensor, reduceIt:bool = True)->tuple[Tensor,Tensor]|tuple[Tensor, LazyTensor]:
    g = setMulCouplings(aHat.dtype, aHat.device)
    def doIt(x,y):
        return x @ g @ y
    enum = enumIt(
        range(aHat.shape[0]),
        range(bHat.shape[0])
    )
    idx = enum.__tensor__()
    Basis:Tensor = vmap(
        vmap(
            doIt
        )
        )(
            aHat[idx[:,0]], 
            bHat[idx[:,1]].to(dtype =aHat.dtype, device = aHat.device)
    )
    _, contracted_idx = unique(
            toR(
                Basis
            ).any(
                dim=-1
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
    Basis = contract(V, Basis, dims = ([0], [0]))
    Coef = contractCoef(aC, bC, idx, enum, V, dims=([0],[0]), out = None)
    if(reduceIt):
        return reduceBasisMat(Basis, Coef)
    else:
        return Basis, Coef
    
def doMulStateMat(aHat:Tensor, aC:Tensor, psiHat:Tensor, psiC:Tensor, reduceIt:bool = True, renorm:bool = False) -> tuple[Tensor, Tensor]|tuple[Tensor,Callable[[Tensor],Tensor]]:
    g = setStateOperCouplings(aHat.dtype, aHat.device)
    def doIt(Opr:Tensor, Psi:Tensor)->Tensor:
        return Psi @ g @ Opr
    enum = enumIt(
        range(aHat.shape[0]),
        range(psiHat.shape[0])
    )
    idx = enum.__tensor__()
    Basis = vmap(
        vmap(
            doIt
        )
        )(
            aHat[idx[:,0]].to(dtype = psiHat.dtype, device = psiHat.device), 
            psiHat[idx[:,1]]
    )
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
    Basis = contract(V, Basis, dims = ([0],[0]))
    Coefs = contractCoef(aC, psiC, idx, enum, V, dims=([0],[0]))
    if(reduceIt):
        return reduceBasisMat(Basis, Coefs)
    else:
        return Basis, Coefs

def doMulStateState(betaHat:Tensor, betaC:Tensor, betaTp:str, alphaHat:Tensor, alphaC:Tensor, alphaTp:str)->Tensor:
    functional = isinstance(LazyTensor)
    assert not functional, NotImplementedError("""Multiplication is not yet implemented for the LazySU2States solution is to \nt=Tensor([...], dtype=..., device=...)\nState1(val:Tensor) @ State2(val:Tensor)\n""")
    match (betaTp, alphaTp):
        case ('ket','bra'):
            idx = enumIt(range(betaHat.shape[0]), range(betaHat.shape[0])).__tensor__()
            Cfs = betaC[idx[:,1]] * alphaC[idx[:,0]]
            return contractCoef(
                Cfs, 
                (
                    alphaHat[idx[:,0]]*betaHat[idx[:,1]]
                ).sum(dim=-1).ceil().all(dim=-1).to(alphaC.dtype)
                ,
                dims = ([0],[0])
            ).sum(
                dim=0
            )
        case ('bra', 'ket', False):
            raise NotImplementedError('Outer Product Not yet Implemented')
            return 
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

def setMulCouplings(dtype:tdp, device:tdevice)->Tensor:
    global mulCouplings
    try:
        mulCouplings = mulCouplings.to(dtype=dtype, device = device)
        return mulCouplings
    except:
        mulCouplings = [eye(4, dtype=dtype, device=device)]
        e = levi_cevita_tensor(3)
        for i in range(1,4):    
            A = zeros(4,4, dtype = dtype, device=device) 
            A[i,0] = 1
            A[0,i] = 1
            for j in range(1,4):
                for k in range(1,4):
                    if(i != j and i != k):
                        if(e[j-1, k-1,i-1]):
                            A[j,k] = (1j*e[j-1,k-1,i-1])
            mulCouplings.append(A)
        mulCouplings = stack(mulCouplings)
        return mulCouplings

def setStateOperCouplings(dtype:tdp, device:tdevice)->Tensor:
    global stateCouplings
    try:
        stateCouplings.to(dtype, device= device)
        return stateCouplings
    except:
        stateCouplings = tensor(
            [
                [
                    [1,0,0,1],
                    [0,1,-1j,0]
                ],
                [
                    [0,1,1j,0],
                    [1,0,0,-1]
                ]
            ],
            dtype= dtype,
            device=device
        )
        return stateCouplings
