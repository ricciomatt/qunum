
import torch

@torch.jit.script
def cummatprod_(O: torch.Tensor, left_or_right:str = 'left') -> torch.Tensor:
    for i in range(1, O.size(0)):
        if(left_or_right == 'left'):
            O[i] = O[i] @ O[i-1]
        else:
            O[i] = O[i-1] @ O[i]
    return O


@torch.jit.script
def matprodcontract_(O:torch.Tensor, left_or_right:str='left')->torch.Tensor:
    V = O[0]
    for i in range(1, O.size(0)):
        if(left_or_right == 'left'):
            V = O[i] @ V
        else:
            V = V @ O[i]
    return V