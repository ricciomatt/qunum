
import torch
@torch.jit.script
def pT_arr(p:torch.Tensor, ixs:torch.Tensor):
    k = torch.empty_like(p)
    for i in range(ixs.shape[0]):
        for j in range(ixs.shape[0]):
            t = [[t for m in range(ixs.shape[1])]
                 for t in ixs[i]]
            l = [ixs[j] for m in range(ixs[j].shape[0])]
            k[t,l] = (p[t,l].T)
    return k
    
