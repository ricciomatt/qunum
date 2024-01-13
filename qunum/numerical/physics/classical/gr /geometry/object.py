import torch
from torch import Tensor
from .functions import metric_grad, christoffel, christoffel_grad, ricci_tensor, riemann_tensor

class EinstienHilbert:
    def __init__(self, g_uv:Tensor, x:Tensor)->Tensor:
        self.g_uv = g_uv
        self.g_UV = torch.inverse(self.g_uv)
        self.x = x
        self.L = None
        self.g_uva = None
        self.R_abcd = None
        self.R_ab = None
        self.R_scalar = None
        return
    
    def G_uv(self):
        if(self.R_scalar is None):
            self.R()
        return self.R_ab - ((1/2) * self.R_scalar*self.g_uv)
    
    def R(self):
        if(self.R_ab is None):
            self.R_uv()
        if(self.R_scalar is None):
            self.R_scalar = (self.R_uv*self.g_UV).sum()
        return self.R_scalar
    
    def R_uv(self):
        if(self.R_abcd is None):
            self.R_uavb()
        if(self.R_ab is None):
            self.R_ab = ricci_tensor(self.R_abcd)
        return self.R_ab
    
    def R_uavb(self):
        if(self.L is None):
            self.Christofell()
        if(self.R_abcd is None):
            self.Lu = christoffel_grad(self.L, self.x)
            self.R_abcd = riemann_tensor(self.L, self.Lu)
        return self.R_abcd
    
    def Christofell(self):
        if(self.L is None):
            self.g_uva = metric_grad(self.g_uv, self.x, grad=True)
            self.L = christoffel(self.g_uva, self.g_UV)
        return self.L
    
        