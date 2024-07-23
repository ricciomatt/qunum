from ......tensors import levi_cevita_tensor
from ......combintorix import EnumUniqueUnorderedIdx as enumIt
from string import ascii_uppercase, ascii_lowercase
from torch import from_numpy, einsum, Tensor, unique, view_as_real as toR, view_as_complex as toC, empty, jit, arange, vmap


def doMul(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor)->tuple[Tensor,Tensor]:
    Basis, idx, temp = mul_basis(
                    aHat, 
                    aC,
                    bHat,
                    bC,
                    levi_cevita_tensor(3, to_tensor=True).to(dtype = aHat.dtype, device= aHat.device), 
                    pfx = ascii_uppercase[:len(aHat.shape)-1],
                    idx=from_numpy(
                        enumIt(
                            range(aHat.shape[0]), 
                            range(bHat.shape[0])
                        ).__array__()
                    )
                )
    V:Tensor = vmap(lambda x: (x==idx).to(temp.dtype),  out_dims=1)(arange(Basis.shape[0]))
    return Basis, einsum('A, Ai->i',temp, V)


@jit.script
def mul_basis(aHat:Tensor, aC:Tensor, bHat:Tensor, bC:Tensor, e:Tensor, pfx:str, idx:Tensor)->tuple[Tensor,Tensor,Tensor]:
    mHat = empty((idx.shape[0], aHat.shape[-2], aHat.shape[-1]), dtype = aHat.dtype, device= aHat.device)
    mHat[...,0] = einsum(f'{pfx}i, {pfx}i->{pfx}', aHat[idx[:,0]], bHat[idx[:,1]])
    mHat[..., 1:] = (
            1j*einsum(f"{pfx}i, {pfx}j, ijk->{pfx}k", aHat[idx[:,0],...,1:], bHat[idx[:,1],...,1:], e) + 
            einsum(f'{pfx}i, {pfx}->{pfx}i', aHat[idx[:,0],...,1:], bHat[idx[:,1],...,0]) + 
            einsum(f'{pfx}i, {pfx}->{pfx}i', bHat[idx[:,1],...,1:], aHat[idx[:,0], ...,0])
        )
    ret:tuple[Tensor, Tensor] = unique(
            toR(
                mHat
            ),
            return_inverse=True, 
            dim=0
        )
    temp:Tensor = aC[idx[:,0]]*bC[idx[:,1]]
    return toC(ret[0]),ret[1], temp
        
def mul_state()->Tensor:
    return 