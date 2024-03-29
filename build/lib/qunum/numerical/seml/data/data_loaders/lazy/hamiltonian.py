import numpy as np 
from numpy.typing import NDArray
import torch 
from typing import Callable

    
class LazyTimeHamiltonian:
    def __init__(self, 
                 hamiltonain:Callable = None,
                 dt:float = 1e-3,
                 num_steps:int = 10,
                )->None:
        self.dt = dt
        self.n = 0
        self.num_steps = num_steps
        self.H = hamiltonain
        return
    
    def __call__(self,
                 t:None|float|list[float]=None, 
                 ix:None|int|list[int] = None,
                 dt:None|float = None,
                 generator:bool=False,
                 )->torch.Tensor|NDArray:
        if(dt is None):
            dt = self.dt
        if(t is None):
            if(ix is None):
                raise Exception('Need to pass index or time of iteration')
            else:
                try:
                    ix[0]
                    t = np.array(ix)*dt 
                except:     
                    t = np.array([ix*dt])      
        else:
            try:
                t[0]
            except:
                t = [t]
        if not generator:
            return self.H(t)
        else:
            return map(self.H,t)
    
    def __iter__(self)->object:
        return self
    
    def __next__(self)->torch.Tensor:
        if self.n < self.num_steps:
            self.n+=1
            return self.H(torch.tensor([self.n])*self.dt)
        else:
            self.num_steps+=self.num_steps
            raise StopIteration
    
    def __getitem__(self, ix:int|list[int]|torch.Tensor):
        if(type(ix) != int):
            try:
                ix.type_as(torch.complex64)
            except:
                ix = torch.tensor(ix, dtype=torch.complex64)
            return self.H(ix*self.dt)
        else:
            return self.H(torch.tensor([ix])*self.dt)
    def to(self,device:str|int)->None:
        try:
            self.H.to(device)
        except:
            pass 
        return
        
    