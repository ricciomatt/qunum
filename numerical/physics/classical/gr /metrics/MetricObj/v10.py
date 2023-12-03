'''v10 Basically we can use this object to compute quantities
given a function for the metric.
Next we should solve the field equations numerically, giving us a solution
for a gerneal energy momentum tensor. For instance given some T_uv find g_uv(x)
from there you can compute all different quantitites.
This will be very difficult, given an energy momentum distribution one can bootstrap
a solution by taking the energy momentum tensor over the space, and evaluating
the einstein tensor.
I have to find a way to generally solve the field equations numerically for this to be viable.
'''
import numpy as np
try:
    import cupy as cp
    from numba import cuda
except Exception as e:
    print(e)
    import numpy as cp
import torch
from numerical.classical.relativistic.gr.geometry import functions as gn
from ..MetricFunctions import cv as mfuncts
from numpy.typing import NDArray
from typing import Callable


class MetricObj:
    def __init__(self, metric_args:dict = None, metric_function:Callable = None,)->object:
        '''
        Pass in a function that defines the metric tensor,
        for instance, this must be in the format
        metric_function(g_init(n, 4, 4) np.array or torch.tensor ,
        x(n, 4) np.array or torch.tensor,
         metric_args dictionary of keyword args,
        grad boolean to tell if we are going to need to compute the gradient of the metric
        with respect to x
        )
        and the metric_args such as r_s for the schwartschild metric.
        you can build any function to call
        '''
        if(metric_function is None):
            metric_function = mfuncts.schwarts
        if(metric_args == None):
            metric_args = {'r_s':100,}
        self.metric_function = metric_function
        self.metric_args = metric_args
        return
    
    def get_g_uv(self, x:NDArray, grad:bool = True)-> tuple[NDArray, NDArray| torch.Tensor, torch.Tensor]:
        if(grad):
            try:
                x = torch.tensor(x, requires_grad=True).type(torch.cuda.FloatStorage)
            except:
                x = torch.tensor(x, requires_grad=True)
            try:
                g_uv = torch.tensor(np.zeros((x.shape[0], 4, 4))).type(torch.cuda.FloatStorage)
            except:
                g_uv = torch.tensor(np.zeros((x.shape[0], 4, 4)))
            g_uv = self.metric_function(g_uv, x, self.metric_args, grad = True)
            return g_uv, x
        else:
            g_uv = np.zeros((x.shape[0], 4, 4))
            g_uv = self.metric_function(g_uv, x, self.metric_args, grad = False)
            return g_uv, x
    
    def get_christ(self, x:NDArray, grad = False)->object:
        g_uv, x = self.get_g_uv(x, grad = True)
        guva = gn.metric_grad(g_uv, x, grad=False)
        if not grad:
            g_UV = np.linalg.inv(g_uv.detach.numpy())
            L = gn.christ_par(guva.detach.numpy(),g_UV)
            return L, g_uv.detach.numpy()
        else:
            g_UV = torch.inverse(g_uv)
            L = gn.christ_torch(guva,g_UV)
            return L, g_uv, x
    
    def R_upva(self,x0:NDArray,) -> tuple[np.array, np.array]:
        L,g,x = self.get_christ(x0, grad=True)
        Lu = gn.christ_grad(L,x)
        R_abcd = gn.Riemann_Tensor(Lu.detach().numpy(), L.detach().numpy())
        return R_abcd, g.detach().numpy()
    
    def R_uv(self, x0:NDArray) -> tuple[np.array, np.array]:
        R_upva,g = self.R_upva(x0)
        R_uv = gn.Ricci_Tensor(R_upva)
        return R_uv, g
    
    def R(self,x0:NDArray) -> tuple[NDArray, NDArray]:
        R_uv, g = self.R_uv(x0)
        R = np.sum(R_uv*np.linalg.inv(g), axis = 1)
        R = np.sum(R, axis = 1)
        return R, g

    def G_uv(self, x0:NDArray)->NDArray:
        R, g = self.R(x0)
        R_uv, g = self.R_uv(x0)
        return R_uv - (g/2)*R

    def geodesic(self,x0:NDArray,
                 xd0:NDArray,
                 m: float,
                 num_steps:int = 100,
                 epsilon:float = 1e-3)->tuple[NDArray, NDArray]:
        x = np.empty((num_steps, x0.shape[0]), dtype=np.float64)
        xd = np.empty((num_steps, x0.shape[0]), dtype=np.float64)
        tau = np.arange(num_steps)
        for i in tau:
            Gamma, g = self.get_christ(x0,grad=False)
            xdd = np.dot(np.dot(Gamma, xd0),xd0)
            xd0 += xdd * epsilon
            x0 += xd0*epsilon + xdd*epsilon**2/2
            x[i,:] = x0
            xd[i,:] = xd0
        tau *= epsilon
        return x, xd
