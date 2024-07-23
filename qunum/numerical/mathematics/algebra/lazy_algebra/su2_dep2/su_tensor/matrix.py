import torch
from ..basis import MatrixBasis
import numpy as np
class TensorSUn:
    def __init__(self, coefficents:torch.Tensor, basis:np.ndarray[MatrixBasis]):
        self.Basis = basis 
        self.Coefficents = coefficents
        return 
    
    def __add__(self)->MatrixBasis:
        pass

    def __call__(self):
        pass