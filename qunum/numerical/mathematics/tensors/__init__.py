import torch
import itertools
import numpy as np
from ...physics.quantum.qobjs import TQobj
def levi_cevita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

# Example usage:

def eye(dim:int, dtype:torch.TypedStorage = torch.complex128, device:str|int = 'cpu', **kwargs)->TQobj:
    return TQobj(torch.eye(dim,dtype=dtype).to(device=device), **kwargs)
