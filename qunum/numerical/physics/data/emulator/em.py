from torch.utils.data import Dataset, DataLoader, Sampler
from torch.distributions import Normal, Distribution
from torch import Tensor, Size
from typing import Tuple
def pinn_sim_data_loader(
        xSim:Tensor,
        ySim:Tensor, 
        batch_size:int|None = 10, 
        shuffle:bool = True,
        batch_sampler:Sampler|None = None,
        **kwargs
    )->DataLoader:
    return DataLoader(
        LazyPinnSimDataSet(ySim, xSim), 
        batch_sampler = batch_sampler, 
        batch_size = batch_size, 
        shuffle = shuffle,
        **kwargs
    )


class SquareNornal(Normal):
    def __init__(self, *args, **kwargs):
        super(SquareNornal, self).__init__(*args, **kwargs)
        
    def rsample(self, n_sample:Size|tuple[int,...])->Tensor:
        return super(SquareNornal, self).rsample(n_sample)**2



class LazyPinnSimDataSet(Dataset):
    def __init__(self, ySim:Tensor, xSim:Tensor, turnSim:None|int = int(1e3), xSampler:Distribution = SquareNornal(0, .1)):
        self.ySim = ySim
        self.xSim = xSim
        self.xSampler = xSampler
        self.HeavisideThetaOff = turnSim
        self.n = 0
        return
    
    def __iter__(self)->object:
        return self 
    
    def __getitem__(self, index) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, None]]|Tuple[Tensor, Tensor]:
        t = self.xSim[index]
        return ((t, self.xSampler.rsample(t.shape)), (self.ySim[index], None))
