import torch
from torch import autograd as AutoGrad
from .....algebra import commutator
from ....data.data_loaders.physics.quantum import LazyTimeHamiltonian
from scipy.special import bernoulli
class MagnusApprx:
    def __init__(self,  
                 H:LazyTimeHamiltonian,
                 N:int,
                 order:int=7):
        self.sp = H[0].shape
        self.H = H
        self.U = torch.eye(self.sp[0], self.sp[1])
        self.order=  order
        self.num_stes = N
        self.n = 0
        self.coef = bernoulli(order)
        return
    
    def __iter__(self)->object:
        return self
    
    def __next__(self)->torch.Tensor:
        if(self.n<self.num_stes):
            Hi = self.H[self.n]
            self.n+=1
            Hip1 = self.H[self.n]
            cor =  (Hi * self.H.dt)
            
            ### Permute the indiceis, compute the Hamiltonian on all
            for o in range(self.order):
                if(self.coef[o] != 0):
                    t = self.coef[o]/torch.math.factorial(o+1)
                    self.U += t*commutator(Hi,Hip1)*self.H.dt**(o+1)
            
            return
        else:
            raise StopIteration
    
    def reset(self):
        self.U = torch.eye()
        
def magnus_function():
    pass



