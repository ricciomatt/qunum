import torch, numpy as np
from torch import einsum
def combine_indicies(m:int = 2, n:int = 2):
    x = torch.tensor(np.array(list(map(lambda x: np.arange(n), np.arange(m)))))
    X_Grid = list(torch.meshgrid(*x))
    X_Grid = [x.flatten(0) for x in X_Grid]
    return torch.stack(tuple(X_Grid)).T

def partial_trace(M:torch.Tensor, trace_ix:int = 0, m:int = 2, n:int = 2):
    ix = combine_indicies(m=m,n=n)
    Tr = torch.zeros((n,n), dtype = M.dtype)
    for i in range(n):
        six = torch.where(ix[:, trace_ix] == i)[0]
        Tr[i,i] += M[six, six].sum()
    return Tr


def get_cols(ix):
    return f"column_{str(ix)}"
vgc = np.vectorize(get_cols)

@torch.jit.script
def ptrace_ix(ix:torch.Tensor, p:torch.Tensor)->torch.Tensor:
    pA = torch.empty(p.shape[0],ix.shape[0], ix.shape[0])
    for i in range(ix.shape[0]):
        for j in range(ix.shape[0]):
            pA[i,j] = p[ix[i], ix[j]].sum()
    return pA
@torch.jit.script
def ptrace_torch(rho:torch.Tensor, trace_ix:int = 0, m:int = 2, n:int = 2):
    return 