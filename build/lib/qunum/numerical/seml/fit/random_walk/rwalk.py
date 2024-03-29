import torch
from typing import Iterator
from torch.distributions import Normal, Uniform, Gamma

class MonteCarloParms(torch.Tensor):
    def __init__(self, *args, num_steps:int = int(1e2), steppers:None|dict=None, default_stepper:Normal|Uniform = Normal(0,.1), step_scale:float = 1e-3):
        super(MonteCarloParms, self).__init__()
        self._metadata = dict(steppers=steppers, default_stepper=default_stepper, epsi = step_scale, num_steps = num_steps, n = 0)
        return
    
    def __iter__(self)->Iterator:
        return self 
    
    def __next__(self,)->None:
        self._metadata['n']+=1
        if(self._metadata['n'] < self._metadata['num_steps']):
            self.step()
            return 
        else:
            raise StopIteration    
        
    
    def __len__(self):
        return self._metadata['num_steps']
    
    def size(self):
        return self.shape
    
    def step(self):
        self += self._metadata['epsi']*self._metadata['default_stepper'].rsample(self.shape)
        return 
    
    def reset_iterator(self):
        self._metadata['n'] = 0
        return