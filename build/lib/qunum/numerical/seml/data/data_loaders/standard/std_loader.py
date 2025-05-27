from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler, RandomSampler, WeightedRandomSampler, Sampler
from torch import Tensor, tensor, float32
import numpy as np 
import torch
from ...pipelines import Pipeline
from typing import Callable

def make_data_loader(x, y, batch_size:int = None, batch_pct:float = .1, pipeline:Pipeline = None, randomize:bool = True, ax_data:int = 0, requires_grad:bool = False, ToTesnor:bool = True):
    if(batch_size is None):
        batch_size = int(batch_pct*x.shape[0])
    return DataLoader(GenDataSet(x,y,pipeline=pipeline, randomize=randomize, ax_data=ax_data, ToTensor=ToTesnor), batch_size=batch_size, shuffle=randomize, requires_grad = requires_grad,)

class GenDataSet(Dataset):
    def __init__(self, x, y, pipeline:Pipeline = None, randomize:bool=True, ax_data:int = 0, requires_grad:bool = False, ToTensor:bool = True):
        super(GenDataSet, self).__init__()
        ixs = np.arange(x.shape[ax_data])
        if(pipeline is None):
            pipeline = Pipeline([])
        if(randomize):
            np.random.shuffle(ixs)
        self.x = x[ixs]
        self.y = y[ixs]
        self.pipeline = pipeline
        self.requires_grad = requires_grad
        self.ToTensor = ToTensor
        return
    
    def __getitem__(self, index:int)->Tensor:
        x = self.x[index]
        y = self.y[index]
        if(self.pipeline is not None):
            x = self.pipeline(self.x[index])
        if(self.ToTensor):
            x = tensor(x, dtype=float32)
            x = x.requires_grad_(self.requires_grad)
            return x, tensor(y, dtype = float32)
        else:
            return x, y
    
    def __len__(self)->int:
        return self.x.shape[0]
    
    def shape(self)->tuple[tuple[int], tuple[int]]:
        return (self.x.shape, self.y.shape)

