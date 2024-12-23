from torch import einsum, Tensor, unique, view_as_real as toR, jit, arange, vmap, empty_like, tensordot as contract, eye, zeros, stack, dtype as tdp, device as tdevice, tensor, where
from .......mathematics.tensors import levi_cevita_tensor
from .......mathematics.combintorix import EnumerateArgCombos as enumIt, EnumUniqueUnorderedIdx as enumUni
from .compress import reduce



def doMul(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor, reduceIt:bool = True)->tuple[Tensor,Tensor]:
    g = setMulCouplings(aHat.dtype, aHat.device)
    def doIt(x,y):
        return x @ g @ y
    idx = enumIt(
        range(aHat.shape[0]),
        range(bHat.shape[0])
    ).__tensor__()
    Basis = vmap(
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
    V:Tensor = vmap(lambda x: (
        x==contracted_idx
    ).to(aC.dtype),  out_dims=1)(arange(Basis.shape[0]))
    
    Basis = contract(V, Basis, dims = ([0], [0]))
    Coefs = (aC[idx[:,0]]*bC[idx[:,1]]) @ V
    if(reduceIt):
        return reduce(Basis, Coefs)
    else:
        return Basis, Coefs



def doMulStateMat(aHat:Tensor, aC:Tensor, psiHat:Tensor, psiC:Tensor, reduceIt:bool = True) -> Tensor:
    g = setStateOperCouplings(aHat.dtype, aHat.device)
    def doIt(Opr:Tensor, Psi:Tensor)->Tensor:
        return Opr @ g @ Psi
    idx = enumIt(
        range(aHat.shape[0]),
        range(psiHat.shape[0])
    ).__tensor__()
    Basis = vmap(
        vmap(
            doIt
        )
        )(
            aHat[idx[:,0]].to(dtype = psiHat.dtype, device = psiHat.device), 
            psiHat[idx[:,1]]
    )
    Coefs = aC[idx[:,0]]*psiC[idx[:,1]]
    _, contracted_idx = unique(
            toR(
                Basis
            ).any(
                dim=-1
            ),
            return_inverse=True, 
            dim=0
        )
    V:Tensor = vmap(lambda x: (x==contracted_idx).to(psiC.dtype), out_dims=1)(arange(Basis.shape[0]))
    return contract(V, Basis, dims = ([0],[0])), Coefs @ V

def doMulStateState(betaHat:Tensor, betaC:Tensor, betaTp:str, alphaHat:Tensor, alphaC:Tensor, alphaTp:str)->Tensor:
    match (betaTp, alphaTp):
        case ('ket','bra'):
            idx = enumIt(range(betaHat.shape[0]), range(betaHat.shape[0])).__tensor__()
            return (((alphaHat[idx[:,0]]*betaHat[idx[:,1]]).sum(dim=-1).ceil().all(dim=-1).to(alphaC.dtype) ) * betaC[idx[:,1]] * alphaC[idx[:,0]]).sum()
        case ('bra', 'ket'):
            raise NotImplementedError('Outer Productd Not yet Implemented')
            return 
        case _:
            raise TypeError('State must be \\bra{\\beta}\\ket{\\alpha} or \\ket{\\beta}\\bra{\\alpha}')
    return

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
