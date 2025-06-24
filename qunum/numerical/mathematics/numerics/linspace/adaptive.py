from typing import Self, Callable
from torch import Tensor
import torch 
from ...tensors.lazy import LazyTensor

from numpy import ndarray
class AdaptiveLinspace:
    def __init__(
        self, dt:float|Tensor, 
        Nsteps:int, 
        t0:float|Tensor|int = 0., 
        adaptive_function:Callable[[Tensor], Tensor]|LazyTensor|None = None, 
        adaptive_epsilon:float|None = None,
        dtype:torch.dtype=torch.float32, 
        method_='f', 
        device:torch.device = 'cpu'
    )->Self:
        self.dt0 = self.__getTensor(dt, dtype = dtype, device = device)
        if(adaptive_epsilon is None): self.epsilon = self.dt0.clone
        else: self.epsilon = self.__getTensor(adaptive_epsilon, dtype = dtype, device = device)
        self.dt = torch.empty((Nsteps), dtype =dtype, device =device)
        self.adaptive_function:LazyTensor|Callable[[Tensor], Tensor]|None = adaptive_function
        self.method_:str = method_
        self.t = torch.empty((Nsteps+1), dtype = dtype, device = device)
        self.t[0] = t0
        self.Nsteps = Nsteps
        self.n = 0
        return
    def __iter__(self)->Self:
        return self
        
    def __getTensor(self:Self, dt:Tensor|float|int|complex, dtype:torch.dtype, device:torch.device)->Tensor:
        match dt:
            case float():
                dt= torch.tensor(dt,dtype = dtype, device = device)
            case int():
                dt= torch.tensor(float(dt),dtype = dtype, device = device)
            case complex():
                dt = torch.tensor(dt, dtype = dtype, device = device)
            case Tensor():
                dt = dt.detach().clone().to(dtype = dtype, device = device)
        return dt
        
    def __next__(self)->tuple[Tensor, Tensor]:
        if(self.n<self.Nsteps):
            match (self.adaptive_function, self.method_):
                case (None, _):
                    self.dt[self.n] = self.dt0.clone()
                    self.t[self.n+1] = self.t[self.n] + self.dt0
                    self.n+=1
                    return self.t[self.n], self.dt
                case (_, 'f'):
                    f = self.adaptive_function(self.t[self.n]).abs()
                    dt = min((1/f * self.epsilon, self.dt0)).detach()
                    self.t[self.n+1] = self.t[self.n] + dt
                    self.dt[self.n] = dt.clone()
                    self.n+=1
                    return self.t[self.n], dt
                case (_, "df"):
                    return 
                case (_,_):
                    raise ValueError('Could Not resolve method for this')
        else:
            raise StopIteration
    
    def __gettimestep(self, ix:int|tuple[int,int]|list|Tensor|ndarray)->Tensor:
        
        match ix:
            case int():
                assert ix<self.Nsteps, IndexError('Index out of bounds')
                while(self.n-1 < ix):
                    next(self)
                return self.dt[ix], self.t[ix]
            case slice()|range():
                if(ix.stop is None):
                    stp = self.Nsteps
                else:
                    stp = ix.stop
                while self.n<stp:
                    next(self)

                if(ix.start is None):
                    st = 0
                else:
                    st = ix.start
                if(ix.step is None):
                    return self.dt[st:stp], self.t[st:stp]
                else:
                    return self.dt[list(ix)], self.t[ix]
            case Tensor():
                while self.n<=ix.max():
                    next(self)
                return self.dt[ix], self.t[ix] 
            case ndarray():
                while self.n<=ix.max():
                    next(self)
                return self.dt[ix], self.t[ix]
            case list()|tuple():
                while self.n<=max(ix):
                    next(self)
                return self.dt[ix], self.t[ix]
                
                
    def __getitem__(self, ix:int|slice|Tensor|list|ndarray|tuple|range)->tuple[Tensor, Tensor]:
        return self.__gettimestep(ix) 
    def __repr__(self)->str:
        return 'AdaptiveLinspace(dt = {dt:.3e}, n = {n}, Nsteps={Nsteps}, method_ = "{method_}")'.format(
            dt = str(self.dt0), n=str(self.n), method_= self.method_, Nsteps = str(self.Nsteps)
        )