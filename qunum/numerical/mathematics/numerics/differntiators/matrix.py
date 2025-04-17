import torch
from typing import Self
from warnings import warn
class Diff:
    def __init__(self, npts:int, dx:float, order:int= 1, method:str='forward', boundary:str = 'open', dtype:torch.dtype = torch.complex128, device:torch.device = 'cpu')->Self:
        self.npts:int = int(npts)
        self.order:int = int(order)
        if(method not in ['forward', 'backward', 'symmetric']):
            warn("method must be in ['forawrd', 'backward', 'symmetric'] but found {met} asserting forward".format(met = method))
            method:str = str('forward')
        if(boundary not in ['open', 'cyclic']):
            warn("boundary must be in ['open', 'cyclic'] but found {met} asserting forward".format(met = boundary))
            boundary:str = str('open')
        self.method:str = str(method)
        self.boundary:str = str(boundary)
        self.device:torch.device = device
        self.dtype:torch.dtype = dtype
        self.dx:float = float(dx)
        return
    
    def __call__(self, dx:float|None = None, method:str = 'forward', boundary:str = 'open')->torch.Tensor:
        if(dx is None):
            dx = self.dx
        if(boundary not in ['open', 'cyclic']):
            warn("boundary must be in ['open', 'cyclic'] but found {met} asserting open".format(met = boundary))
            boundary:str = str('open')
        match method:
            case 'forward':
                D = self.getForward(boundary=boundary)
            case 'backward':
                D = self.getBackward(boundary=boundary)
            case 'symmetric':
                D = self.getSymmetric(boundary=boundary)
            case _:
                raise ValueError("method must be in ['forawrd', 'backward', 'symmetric'] but found {met} asserting forward".format(met = method))
        return torch.matrix_power(D/dx, self.order)
    def getForward(self,  boundary:str = 'open')->torch.Tensor:
        D= torch.diag(
            torch.ones(self.npts - 1, dtype=self.dtype, device=self.device), 
            diagonal=1
        ) - torch.diag(
            torch.ones(self.npts, dtype=self.dtype, device=self.device)
        )
        if(boundary == 'cyclic'):
            D[-1,0] = 1
        return D
    def getBackward(self,  boundary:str = 'open')->torch.Tensor:
        D = -1*torch.diag(
            torch.ones(self.npts - 1, dtype=self.dtype, device=self.device), 
            diagonal=-1
        ) + torch.diag(
            torch.ones(self.npts, dtype=self.dtype, device=self.device)
        )
        if(boundary == 'cyclic'):
            D[0,-1] = -1
        return D
    def getSymmetric(self,  boundary:str = 'open')->torch.Tensor:
        D =  (-1*torch.diag(
            torch.ones(self.npts - 1, dtype=self.dtype, device=self.device), 
            diagonal=-1
        ) + torch.diag(
            torch.ones(self.npts-1, dtype=self.dtype, device=self.device),
            diagonal = 1
        ))/2
        if(boundary == 'cyclic'):
            D[-1,0] = 1/2
            D[0,-1] = -1/2
        
        return D
    
