from torch.nn import Module, Sequential, MSELoss
from torch import exp, Tensor
import warnings 
try:
    from complexPyTorch.complexLayers import ComplexReLU, ComplexLinear
except:
    warnings.warn('Complex PyTorch not installed, Limited Access to Complex Nueral Networks')
class PsiNetwork(Module):
    def __init__(self, loss_function=MSELoss()):
        super(PsiNetwork, self).__init__()
        self.re = Sequential(    
            ComplexLinear(4, 10),
            ComplexReLU(),
            
            ComplexLinear(10,20),
            ComplexReLU(),
            
            ComplexLinear(20,10),
            ComplexReLU(),
            
            ComplexLinear(10, 2),
        )
        self.loss_function = loss_function
        return 
    def __call__(self ,x:Tensor)->Tensor:
        return exp(self.re(x))
    
    def forward(self, x):
        return self.call(x)
