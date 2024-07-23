
import itertools
import numpy as np
import torch
def levi_cevita_tensor(dim, to_tensor:bool = False )->np.ndarray|torch.Tensor:   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    if(to_tensor):
        return torch.from_numpy(arr)
    else:
        return arr

# Example usage:
