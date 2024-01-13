import torch 
from torch import Tensor
from ...data import DataLoader, LazyLattice
from ...metrics.numerical import ModelTracker
from tqdm import tqdm
from typing import Callable
from ...metrics.numerical import mean_accuracy
import numpy as np
from copy import deepcopy
class PhysicsDataGenerator:
    def __init__(self) -> None:
        pass

#Needs Work
class GradDescentTrain:
    def __init__(self, 
                 Model:torch.nn.Module,
                 Loss:Callable|tuple[Callable],
                 dataLoader:DataLoader|LazyLattice|PhysicsDataGenerator,
                 Optimizers:torch.optim.Optimizer|tuple[torch.optim.Optimizer],
                 epochs:int = int(1e1), 
                 batch_steps:int= 1,
                 device:int|str = 0, 
                 validFreq:int = 10,
                 prnt_:bool=False,
                 prntFreq:int = 100,
                 modelTracker:ModelTracker|None=None,
                 dataLoaderValid:DataLoader|LazyLattice|PhysicsDataGenerator|None = None,
                 validationUpdate:Callable|None = None)->None:
        #Model and Loss stuff
        self.Model = Model
        self.Loss = Loss
        
        #dataLoaderStuff
        self.dataLoader = dataLoader
        #Validation Stuff
        self.dataLodaerValid = dataLoaderValid
        self.validationUpdate = validationUpdate
        self.validFreq = validFreq
        
        #Training stuff 
        self.Optimizers = Optimizers
        self.epochs = epochs
        self.batch_steps = batch_steps
        self.device = device
        self.modelTracker = modelTracker
        
        #initialization
        self.tot_epochs = 0
        self.n = 0
        self.tot_training_loops = 0
        self.o = 0
        
        #booleans 
        self.prnt_ = prnt_ 
        self.prntFreq = prntFreq       
        self.opt_iter = is_iterable(self.Optimizers)
        self.track = bool(modelTracker is None)
        self.validate = bool(dataLoaderValid is not None)
        if(self.opt_iter):
            self.O = len(Optimizers)
            step_opt = 'stepopt'
        else:
            self.O = 1
            step_opt = 'stepnormal'
       
        self.step_function = getattr(self, step_opt)
        return 
    
    def reset_iterator(self):
        self.n = 0
        return
    
    def copyModel(self):
        return deepcopy(self.Model)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if(self.n<self.epochs):
            self.n += 1
            self.tot_epochs += 1
            for i in range(self.O):
                self.do_epoch()
                self.o+=1
            self.o = 0
            return 
        else:
            self.tot_training_loops += 1
            raise StopIteration
    
    def __len__(self):
        return (self.epochs, self.batch_steps)

    # Training Looop Logic
    def do_epoch(self):
        self.do_batch()
        if(self.validate):
            if(self.n%self.validFreq == 0): 
                self.do_validate()
        if(self.prnt_):
            if(self.n%self.prntFreq == 0):
                print(self.modelTracker)
        return
    
    def do_validate(self)->None:
        return 

    def do_batch(self)->None:
        for step, (x,y) in enumerate(self.dataLoader):
            x = x.to(self.device)
            y = y.to(self.device)
            for i in range(self.batch_steps):
                self.step_function(x,y)
        return
    
    def train_model(self)->None:
        for i in tqdm(range(self.epochs)):
            next(self)
        self.reset_iterator()
        return
    
    #Loss Stuff
    def eval_loss(self, x:Tensor, y:Tensor)->Tensor:
        yh = self.Model.forward(x,y)
        L = self.Loss(yh, y, x)
        if(self.track):
            self.modelTracker(y,yh,L)
        return L
    
    def closure(self, x:Tensor, y:Tensor)->Tensor:
        self.Optimizers.zero_grad()
        return self.eval_loss(x, y)
    
    #Optimization
    def stepnormal(self,x:Tensor,y:Tensor)->None:
        self.Optimizers.step(lambda: self.closure(x,y))
        return 
    
    def stepopt(self, x:Tensor, y:Tensor):
        return self.Optimizers[self.o].step(lambda: self.closure(x,y))
    
    
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False 