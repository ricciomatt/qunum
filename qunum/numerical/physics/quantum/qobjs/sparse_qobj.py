import torch
from torch import Tensor
class SparseQobj:
    def __init__(self, data:Tensor):
        self.data = data.to_sparse()

        pass