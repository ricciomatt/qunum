import torch
from torch import Tensor
from .meta import QobjMeta
class SparseTQobj:
    def __init__(self, data:Tensor, meta:QobjMeta):
        self.data = data.to_sparse()
        pass