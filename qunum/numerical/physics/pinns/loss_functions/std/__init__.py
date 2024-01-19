from ....quantum.qobjs import TQobj
from torch import prod, tensor
import numpy as np 
class ComplexMSE:
    def __init__(self)->None:
        return
    def __call__(self, y:TQobj, yh:TQobj, *args, **kwargs)->TQobj:
        l = y - yh
        return (l.dag() @ l).sum()/prod(tensor(l.shape[:-2]))