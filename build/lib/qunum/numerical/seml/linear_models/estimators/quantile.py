from torch import Tensor, max
class QuantileLoss:
    def __init__(self, *args, quantile:float = .5, lam:None|float = None, **kwargs)-> None:
        self.quantile = quantile
        if(lam is not None):
            self.quantile = lam
        return
    def __call__(self, yh:Tensor, y:Tensor)->Tensor:
        errors = yh - y
        loss = max((self.quantile - 1) * errors, self.quantile * errors)
        return loss.mean()