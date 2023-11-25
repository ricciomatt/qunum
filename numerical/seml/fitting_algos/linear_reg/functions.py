import numpy as np
import torch
def Id(x:torch.Tensor|np.ndarray):
    return x


def sigmoid(x:torch.Tensor|np.ndarray):
    return 1/(1+torch.exp(x))

def sigmoidI(x:torch.Tensor|np.ndarray):
    return torch.log(1/x - 1)


def tanh(x:torch.Tensor|np.ndarray):
    return torch.tanh(x)

def tanhI(x:torch.Tensor|np.ndarray):
    return torch.atanh(x)



def ptanh(x:torch.Tensor|np.ndarray):
    return (torch.tanh(x)+1)/2

def ptanh(x:torch.Tensor|np.ndarray):
    return torch.atanh(2*x-1)


def exp(x:torch.Tensor|np.ndarray):
    return torch.exp(x)

def expI(x:torch.Tensor|np.ndarray):
    return torch.log(x)
