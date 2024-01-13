import torch
def mean_accuracy(yh:torch.Tensor, y:torch.Tensor):
    return torch.mean((yh-y)/y)
