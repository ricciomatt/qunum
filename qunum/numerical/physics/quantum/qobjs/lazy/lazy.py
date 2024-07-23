from ..meta import QobjMeta
from typing import Self, Callable
from..dense import TQobj
import torch

class LazyQuantumObject:
    def __init__(
            self, 
            GeneratingFunction:Callable[[torch.Tensor|TQobj], TQobj], 
            GenerateInput:torch.distributions.Distribution, 
            **kwargs
        )->Self:
        self.GeneratingFunction = GeneratingFunction
        self.Operations = []
        return
    