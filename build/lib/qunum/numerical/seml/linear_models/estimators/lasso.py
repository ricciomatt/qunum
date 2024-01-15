from typing import Any
from torch import nn, abs, Tensor
from ..models.linear_model import LinearNN
class LassoLoss(nn.MSELoss):
    def __init__(self, *args, lam:float=1., mod:LinearNN=None, **kwargs)->None:
        super(LassoLoss, self).__init__(*args, **kwargs)
        self.mod = mod
        self.lam = lam 
        return 
    def __call__(self, yh:Tensor, y:Tensor, *args, **kwargs) -> Tensor:
        params = self.lam*sum(map(lambda x: abs(x).sum(), set(self.mod.linear.parameters())))
        return params+super(LassoLoss, self).__call__(yh, y)