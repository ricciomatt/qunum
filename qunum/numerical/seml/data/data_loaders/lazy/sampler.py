import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler, RandomSampler, WeightedRandomSampler, Sampler
from torch.distributions import Uniform, Normal, Distribution
from ....distributions.normal.positive_def import SquareNormal
from torch import Tensor
class PhysikLoader:
    def __init__():
        pass

# Need to add data type to initialization of the Lattice Dist

class LazySampler:
    def __init__(
            self, 
            Dists:tuple[Distribution, ...], 
            shps:tuple[int,...]=(1,), 
            requires_grad:bool = True, 
            num_batches:int = int(1e2), 
            batch_size:int = int(1e2)
        )->None:
        self.Dist = Dists
        self.req = requires_grad
        self.num_batches = num_batches
        self.shps = shps
        self.batch_size = batch_size
        self.n = 0
        return
    
    def sampleit(self, size)->tuple[Tensor, None]|tuple[Tensor,Tensor]:
        try:
            return (
                torch.concat(
                    tuple(
                        map(
                            lambda D: D[0].rsample((size, D[1])), 
                            zip(self.Dist, self.shps)
                        )
                    ), 
                    1
                ).requires_grad_(self.req),
                None
            )
        except Exception as e:
            print(e)
            return (
                    self.Dist[0].rsample((size, self.shps[0])).requires_grad_(self.req), 
                    None
                )
    
    def __iter__(self)->None:
        return self
    
    def __getitem__(self, ix:int)->tuple[Tensor, None]|tuple[Tensor,Tensor]:
        if(isinstance(ix,int)):
            return self.sampleit(
                1
            )
        else:
            return self.sampleit(
                len(ix)
            )
    
    def __next__(self)->tuple[Tensor, None]|tuple[Tensor,Tensor]:
        if(self.n < self.num_batches):
            self.n+=1
            return self.sampleit(self.batch_size)
        else:
            self.n = 0
            raise StopIteration
            
    def __len__(self)->int:
        return self.num_batches
    
def lazy_sampler_init(
        Dists:tuple[Distribution, ...]=(SquareNormal(0,1),),
        shps:tuple[int, ...] = (1,),
        batch_size:int = 100, 
        requires_grad:bool = True,
        num_batches:int = int(1e2),
        **kwargs)->LazySampler:
    return LazySampler(
            Dists = Dists, 
            shps = shps, 
            requires_grad = requires_grad, 
            num_batches = int(num_batches),
            batch_size=batch_size
        ) 


class LazyLatticeLoader:
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
    
    def __iter__(self):
        return self
    
    def __next__(self):
        pass

    def __len__(self)-> int:
        return self.num



def lazy_sampler(batch_size:int, stop_iter:int, Samplers:list[Distribution] = [Uniform(0, 1)], requires_grad:bool = True):
    num = 0
    while(True):
        vals = torch.empty((batch_size, len(Samplers)))
        for i, Sample in enumerate(Samplers):
            vals[:, i] = Sample.sample((batch_size,))
        vals = vals.requires_grad_(requires_grad)
        yield vals  
        num+=1
        if(i<stop_iter):
            break
        