from typing import Self, Callable, Iterable
from torch import Tensor, tensordot as contract, device as torchDevice, dtype as torchDtype, einsum, tensor as createTensor, complex128, Size
from .......mathematics.combintorix import EnumerateArgCombos
from .......mathematics.tensors.lazy import LazyTensor, einsumLazy, contractLazy
class State:
    def __init__(self, objTp:str = 'ket', renorm:bool = False) -> Self:
        self.objTp:str= objTp
        self.renorm:bool = renorm
        return
    def set_obj_tp(self, objTp:str='ket')->Self:
        self.objTp = objTp
        return
