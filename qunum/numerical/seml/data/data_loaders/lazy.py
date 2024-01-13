import torch
from torch.distributions import Uniform, Normal
from torch import Tensor
class PhysikLoader:
    def __init__():
        pass

    
class LazyLattice:
    def __init__(self, num_sample:int = 100,  Samplers:list[object] = [Uniform(0, 1)], requires_grad:bool = True):
        self.Samplers = Samplers
        self.num_sample = num_sample
        self.num = 0
        self.requires_grad = requires_grad
        pass 
    
    def __getitem__(self, index:int)->Tensor:
        vals = torch.empty((self.num_sample, len(self.Samplers)))
        for i, Sample in enumerate(self.Samplers):
            vals[:, i] = Sample.sample((self.num_sample,))   
        self.num+=1
        vals = vals.requires_grad_(self.requires_grad)
        return vals
        
    def __len__(self)-> int:
        return self.num



def lazy_sample(num_sample:int, Samplers:list[object] = [Uniform(0, 1)], requires_grad:bool = True):
    num = 0
    while(True):
        vals = torch.empty((num_sample, len(Samplers)))
        for i, Sample in enumerate(Samplers):
            vals[:, i] = Sample.sample((num_sample,))
        vals = vals.requires_grad_(requires_grad)
        yield vals  
        num+=1
        