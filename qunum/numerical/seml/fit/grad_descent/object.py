import torch 
from torch import Tensor
from ...data import DataLoader
from ...metrics.numerical import ModelTracker
from tqdm import tqdm
from typing import Callable, Any
from ...metrics.numerical import mean_accuracy
import numpy as np
from copy import deepcopy
class PhysicsDataGenerator:
    def __init__(self) -> None:
        pass
OptimClosureMap = {torch.optim.LBFGS:True}

class GradDescentTrain:
    def __init__(self, 
                 Model:torch.nn.Module,
                 Loss:Callable,
                 Optimizers:torch.optim.Optimizer|tuple[torch.optim.Optimizer],
                 epochs:int = int(1e1), 
                 batch_steps:int= 1,
                 device:int|str = 0, 
                 validFreq:int = 10,
                 prnt_:bool=False,
                 prntFreq:int = 100,
                 modelTracker:ModelTracker|None=None,
                 dataLoader:DataLoader|PhysicsDataGenerator|None = None,
                 dataLoaderValid:DataLoader|PhysicsDataGenerator|None = None,
                 validationUpdate:Callable|None = None,
                 loss_args:tuple[str] = None,
                 stop_threshold:float|None = None
                 )->None:
        
        #Model and Loss stuff
        self.Model = Model
        self.Loss = Loss
        self.stop_threshold = stop_threshold
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
        
        #initialization
        self.tot_epochs = 0
        self.n = 0
        self.tot_training_loops = 0
        self.o = 0
        
        #booleans 
        self.prnt_ = prnt_ 
        self.prntFreq = prntFreq       
        self.track = bool(modelTracker is None)
        self.validate = bool(validationUpdate is not None)
        
        if(loss_args is not None): 
            self.loss_args = loss_args
        else: 
            from inspect import getfullargspec
            args_map = {'input':'yh', 'target':'y'}
            try:
                self.loss_args = (
                    A 
                        if(A not in args_map) 
                        else 
                    args_map[A] 
                    for 
                        A in getfullargspec(self.Loss.forward).args
                        if 
                            A != 'self'
                )
            except:
                
                self.loss_args = (
                    A 
                        if(A not in args_map) 
                        else 
                    args_map[A] 
                    for 
                        A in getfullargspec(self.Loss).args
                        if 
                            A != 'self'
                )
        if(is_iterable(Optimizers)): 
            self.Optimizers = Optimizers
            self.use_closure = tuple(map(lambda o: OptimClosureMap[type(o)] if(type(o) in OptimClosureMap) else False, self.Optimizers))
        else: 
            self.Optimizers = (Optimizers,)
            self.use_closure = tuple(map(lambda o: OptimClosureMap[type(o)] if(type(o) in OptimClosureMap) else False, self.Optimizers))
        if(modelTracker is not None):
            self.modelTracker = modelTracker
        else:
            self.modelTracker = ModelTracker({}, track_loss=True)
        return 
    
    def reset_iterator(self)->None:
        self.n = 0
        return
    
    def copyModel(self)->torch.nn.Module:
        return deepcopy(self.Model)
    
    def __iter__(self)->None:
        return self
    
    def set_dataLoader(
            self, 
            TrainDataLoader:DataLoader|PhysicsDataGenerator|None
        )->None:
        if(TrainDataLoader is None and self.dataLoader is None):
            raise ValueError('Setting None to None')
        self.dataLoader = TrainDataLoader
        return 

    def __next__(self)->None:
        if(self.dataLoader is None):
            raise ValueError('To iterate provide a data loader with: GD.set_dataLoader(TrainDataLoader)')
        if(self.n<self.epochs):
            self.n += 1
            self.tot_epochs += 1
            for i in range(len(self.Optimizers)):
                self.do_epoch()
                self.o+=1
            self.o = 0
            return 
        else:
            self.tot_training_loops += 1
            raise StopIteration
    
    def __len__(self)->tuple[int,int]:
        return (self.epochs, self.batch_steps)

    # Training Looop Logic
    def do_epoch(self)->None:
        self.do_batch()
        if(self.validate):
            if(self.n%self.validFreq == 0): 
                self.do_validate()
        if(self.prnt_):
            if(self.n%self.prntFreq == 0):
                print(self.modelTracker)
        return
    
    def do_validate(self)->None:
        return self.validationUpdate(self.dataLodaerValid)

    def do_batch(self)->None:
        for step, (x,y) in enumerate(self.dataLoader):
            x = x.to(self.device)
            if(y is not None):
                y = y.to(self.device)
            for i in range(self.batch_steps):
                self.step_function(x,y)
            x.cpu()
            if(y is not None):
                y.cpu()
            del x; del y
        return
    
    def train(self, 
              *args, 
              TrainDataLoader:DataLoader|PhysicsDataGenerator|None = None, 
              ValidationDataLoader:DataLoader|PhysicsDataGenerator|None = None, 
              dropdataLoader:bool= False, 
              epochs:int|None = None,
              stop_threshold:int|None = None,
              **kwargs:dict[Any])->ModelTracker:
        self.Model.to(self.device)
        if(self.dataLoader is None):
            try:
                self.set_dataLoader(TrainDataLoader)
            except:
                if(args == ()):
                    raise ValueError('To train provide a data loader with: GD.set_dataLoader(TrainDataLoader)')
                else:
                    self.set_dataLoader(args[0])
        if(ValidationDataLoader is not None):
            self.dataLodaerValid = ValidationDataLoader
        elif(len(args)>1):
            self.dataLodaerValid = args[1]
        if(epochs is not None):
            self.epochs = epochs
        if(stop_threshold is not None):
            self.stop_threshold = stop_threshold
        
        self.reset_iterator()
        for i,j in tqdm(enumerate(self)):
            pass
        if(dropdataLoader):
            self.dataLoader = None
            self.dataLodaerValid = None 
        self.Model.to('cpu')
        return self.modelTracker
    
    def __call__(self, **kwargs:dict[Any])->None:
        return self.train(**kwargs)

    def getModel(self)->Callable:
        return self.Model
    
    def getTrainingStats(self)->ModelTracker:
        return self.modelTracker   
     
    def getDict(self)->dict:
        return {'Model':self.getModel(), 'modelTracker':self.getTrainingStats(), 'LossFunction':self.Loss}

    
    #Loss Stuff
    def eval_loss(self, **kwargs:dict[Any])->Tensor:
        if('yh' in self.loss_args):
            kwargs['yh'] = self.Model.forward(kwargs['x'])
        else:
            kwargs['Model'] = self.Model
        args = (kwargs[a] for i, a in enumerate(self.loss_args))
        A = self.Loss(*args)
        if(isinstance(A, tuple)):
            L = A[0]
            yh = A[1]
        else:
            L = A
            try:
                yh = kwargs['yh']
            except:
                yh = None
        if(self.track):
            self.modelTracker(kwargs['y'],  yh, L)
        if(self.stop_threshold is not None):
            if(self.stop_threshold>=L):
                self.stop_iteration()
        return L
    
    def stop_iteration(self)->None:
        raise StopIteration

    def closure(self, x:Tensor, y:Tensor)->Tensor:
        self.Optimizers[self.o].zero_grad()
        return self.eval_loss(x = x, y = y)
    
    #Steps the= Parameters for a given optimizer
    def step_function(self,x:Tensor|tuple[Tensor, ...], y:Tensor|tuple[Tensor, ...])->None:
        if(self.use_closure[self.o]):
            L = lambda: self.eval_loss(x = x, y = y)
            with torch.no_grad():
                self.Optimizers[self.o].step(L)
                self.Optimizers[self.o].zero_grad()
        else:
            L = self.eval_loss(x = x, y = y)
            L.backward()
            with torch.no_grad():
                self.Optimizers[self.o].step()
                self.Optimizers[self.o].zero_grad()
        
        return
    
    
def is_iterable(O):
    try:
        iter(O)
        return True
    except:
        return False