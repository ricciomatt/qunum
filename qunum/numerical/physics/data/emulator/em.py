from torch.utils.data import Dataset, DataLoader, Sampler
from torch.distributions import Normal, Distribution
from torch import Tensor, Size
from typing import Tuple, Iterable
import numpy as np 
from torch import empty_like
def pinn_sim_data_loader(
        xSim:Tensor,
        ySim:Tensor, 
        batch_size:int|None = 10, 
        shuffle:bool = True,
        requires_grad:bool = True,
        batch_sampler:Sampler|None = None,
        **kwargs
    )->Iterable:
    return LazyPinnSimDataSet(DataLoader(
        PinnDataSet(ySim, xSim,), 
            batch_sampler = batch_sampler, 
            batch_size = batch_size, 
            shuffle = shuffle,
            **kwargs
        ), 
        req_grad= requires_grad,
    )


class SquareNornal(Normal):
    def __init__(self, *args, **kwargs):
        super(SquareNornal, self).__init__(*args, **kwargs)
        
    def rsample(self, n_sample:Size|tuple[int,...])->Tensor:
        return super(SquareNornal, self).rsample(n_sample)**2


class PinnDataSet(Dataset):
    def __init__(self, 
                 ySim:Tensor, 
                 xSim:Tensor, ):
        self.ySim = ySim
        self.xSim = xSim
        return
    
    def __iter__(self)->object:
        return self 
    
    def __getitem__(self, index) ->Tuple[Tensor, Tensor]:
        t = self.xSim[index]
        return (self.xSim[index]), (self.ySim[index])
    
    def __len__(self, ):
        return self.xSim.shape[0]
    
class LazyPinnSimDataSet:
    def __init__(self,
                 Data:DataLoader,
                 turnSim:None|int = None,
                 xSampler:Distribution = SquareNornal(0, .1),
                 req_grad:bool = False):
        self.Data = Data
        self.xSampler = xSampler
        self.C_iter = None
        self.N = 0
        self.n = 0
        self.HeavisideThetaOff = turnSim
        self.req_grad = req_grad
        return
    
    def __iter__(self)->object:
        return self 
    
    def __next__(self)->Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, None]]:
        if(self.C_iter is None):
            self.C_iter = self.Data.__iter__()
        if(self.n<self.Data.__len__()):
            self.n+=1
            if(self.HeavisideThetaOff is not None):
                if(self.HeavisideThetaOff>self.N):
                    xSim, ySim = next(self.C_iter)
                    X_ = self.xSampler.rsample(xSim.shape)
                else:
                    xSim = None 
                    ySim = None
                    X_ = self.xSampler.rsample(self.Data.batch_size)
            else:
                xSim, ySim = next(self.C_iter)
                X_ = self.xSampler.rsample(xSim.shape)
            X_.requires_grad_(self.req_grad)
            if(xSim is not None):
                xSim.requires_grad_(self.req_grad)
            return ((xSim, X_), (ySim, None))
        else:
            self.n = 0
            raise StopIteration
   
    def turnOff(self)->None:
        self.HeavisideThetaOff = -1 
        return
    
    def __len__(self, ):
        return self.ixs.shape[0]