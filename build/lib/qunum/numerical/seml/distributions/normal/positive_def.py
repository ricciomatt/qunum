from torch import Tensor, Size
from torch.distributions import Normal
class SquareNormal(Normal):
    def __init__(self, *args, **kwargs):
        super(SquareNormal, self).__init__(*args, **kwargs)
        
    def rsample(self, n_sample:Size|tuple[int,...])->Tensor:
        return super(SquareNormal, self).rsample(n_sample)**2
