import torch 
from ...data import DataLoader
from ...metrics.numerical import ModelTracker
from tqdm import tqdm
from typing import Callable
from ...metrics.numerical import mean_accuracy
import numpy as np
class PhysicsDataGenerator:
    def __init__(self) -> None:
        pass

def grad_descent(Model:torch.nn.Module,
                 dataLoader:DataLoader|PhysicsDataGenerator,
                 Optimizer:torch.optim.Optimizer,
                 epochs:int = int(1e1), 
                 batch_steps:int= 1,
                 device:int|str = 0, 
                 prnt_:bool=False,
                 modelTracker:ModelTracker|None=None):
    track_metrics = bool(modelTracker is not None)
    for epoch in tqdm(range(epochs)):
        for step, (x,y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)
            for i in range(batch_steps):
                Optimizer.zero_grad()
                yh = Model.forward(x)
                L = Model.loss(yh, y, x)
                if(prnt_):
                    print(f"{epoch}/{epochs}: Loss={L} {step}/{len(dataLoader)}")
                L.backward()
                with torch.no_grad():
                    Optimizer.step()
                    Optimizer.zero_grad()
                if(track_metrics):
                    modelTracker(y,yh, L)
                yh.detach().cpu()
                del yh                   
            x.detach().cpu(); y.detach().cpu()
            del x; del y
    return Model, modelTracker
