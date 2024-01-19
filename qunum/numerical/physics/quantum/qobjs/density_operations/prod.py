
import torch

@torch.jit.script
def cummatprod_(O: torch.Tensor) -> torch.Tensor:
    for i in range(1, O.size(0)):
        O[i] = O[i] @ O[i-1]
    return O
