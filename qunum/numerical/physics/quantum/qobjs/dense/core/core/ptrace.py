import torch


'''
@nb.njit('complex128[:,:,:](int64[:,:], complex128[:,:,:])', parallel = True, fastmath = True)
def nb_ptrace_ix(ix:NDArray[np.int64], p:NDArray)->NDArray:
    pA = np.empty((p.shape[0],ix.shape[0], ix.shape[0]), dtype = p.dtype)
    for i in nb.prange(ix.shape[0]):
        for j in nb.prange(ix.shape[0]):
            pA[:, i, j] = p[:, ix[i], ix[j]].sum()
    return pA
'''
@torch.jit.script
def ptrace_torch_ix(ix:torch.Tensor, p:torch.Tensor)->torch.Tensor:
    if(len(p.shape) == 2):
        pA = torch.zeros(ix.shape[0], ix.shape[0], dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[i,j] += p[ix[i], ix[j]].sum()
    else:
        pA = torch.zeros(p.shape[0], ix.shape[0], ix.shape[0], dtype = p.dtype)
        for i in range(ix.shape[0]):
            for j in range(ix.shape[0]):
                pA[:, i,j] += p[:, ix[i], ix[j]].sum(dim = [1])
    return pA

