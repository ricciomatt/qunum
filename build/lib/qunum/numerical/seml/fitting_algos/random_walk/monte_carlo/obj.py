from torch.distributions import Normal, Uniform, Gamma
from torch import Tensor, tensor
from numpy import ndarray
import numpy as np 
from typing import Callable
from copy import deepcopy

class MonteCarlo:
    def __init__(self, 
                 Model:Callable,
                 num_threads:int = 10, 
                 num_steps:int = int(1e3), 
                 step_sigma:float = 1e-2, 
                 step_funct:list[Callable]|None = None,
                 Temp:list|ndarray|Tensor|None= None, 
                 maxTemp:float = 1,
                 
                 ):
        if(step_funct is None):
            self.stepper =[Normal(0, step_sigma) for i in Model.parms]
        self.num_threads = num_threads
        self.num_steps = num_steps
        self.Models = [deepcopy(Model) for i in np.arange(self.num_threads)]
        self.num_params = Model.parms.shape[0]
        if(Temp is None):
            Temp = maxTemp*Uniform(0,1).rsample((self.num_threads,)).numpy()
        self.Temp = Temp
        self.n = 0
        self.cost_arrays = np.empty((self.num_threads, self.num_threads))
        self.p_step = np.empty((self.num_threads, self.num_params, self.num_threads))
        return
    
    def step_eval(self):
        if(self.n == 0):
            last_cost = np.array([mod.cost(mod(mod.parms)) for i, mod in enumerate(self.Models)])
            min_cost_parms = [np.min(last_cost), self.Models[np.argmin(last_cost)].parms]
            
        for i, mod in enumerate(self.Models):
            try:
                t = mod.parms.copy()
                t += t*vcstep(self.stepper)
            except:
                t = mod.parms.clone()
                t += t*tensor(vcstep(self.stepper))
            cost = mod.cost(mod(t))
            if(last_cost[i]*np.random.random()*self.Temp[i]>=cost):
                self.Models[i].update_parms(t)
                last_cost[i] = cost
            if(min_cost_parms[0]> cost):
                min_cost_parms[0] = cost; min_cost_parms[1] = deepcopy(t)
            self.cost_arrays[i, self.n] = last_cost[i]
            self.p_step[i, :, self.n] = mod.parms
        return
    
    
    def reset_iterator(self)->None:
        self.n = 0
        return 
    def __iter__(self)->object:
        return self
    
    def __next__(self, ):
        if(self.n<self.num_steps):
            self.step_eval()
            self.n+=1
            return
        else:
            raise StopIteration
def call_step(stepper):
    return stepper.sample().numpy()

vcstep = np.vectorize(call_step)