import torch
from torch import Tensor
from torch import autograd as AutoGrad
import numpy as np
from numpy.typing import NDArray


@torch.jit.script
def metric_grad(g:Tensor, x:Tensor, grad:bool = False)->Tensor:
    
    guva = torch.zeros((x.shape[0], x.shape[1], g.shape[0], g.shape[1]),
                        dtype=torch.float64, requires_grad=grad)
    
    for i in range(g.shape[1]):
        for j in range(g.shape[2]):
            
            g[:, i, j].backward(torch.ones_like(g[:,i,j]),retain_graph=True)
            gx = AutoGrad.grad(g[:,i,j], x, grad_outputs=torch.ones_like(x), allow_unused=True, create_graph=True, retain_graph=True)[0]
            if(gx is None):
                gx = torch.zeros_like(x)
            guva[:,:,i,j] = gx
    return guva


@torch.jit.script
def christoffel(guva:Tensor, g_UV:Tensor)->Tensor:
    try:
        L_uvp = torch.empty(guva.shape, dtype=torch.float32).to(0)
    except:
        L_uvp = torch.empty(guva.shape, dtype=torch.float32)
    for A in range(guva.shape[0]):
        for u in range(guva.shape[1]):
            for v in range(guva.shape[1]):
                for p in range(g_UV.shape[1]):
                    L_uvp[A,u,v,p] = 1/2* torch.dot(g_UV[A, p, :], (guva[A, u,:,v]+guva[A, u,:,v] - guva[A, u,v,:]))
    return L_uvp



@torch.jit.script
def christoffel_grad(L:Tensor, x:Tensor,)->Tensor:
    try:
        Lu = torch.zeros((x.shape[0], x.shape[1], L.shape[1], L.shape[2], L.shape[3]),
                       dtype=torch.float32).to(0)
    except:
        Lu = torch.zeros((x.shape[0], x.shape[1], L.shape[1], L.shape[2], L.shape[3]),
                       dtype=torch.float32)
    for i in range(L.shape[1]):
        for j in range(L.shape[2]):
            for k in range(L.shape[3]):
                L[:,i, j, k].backward(torch.ones_like(L[:,i,j,k]), retain_graph=True)
                Lx = AutoGrad.grad(L[:,i,j,k], x, grad_outputs=torch.ones_like(x), allow_unused=True, create_graph=True, retain_graph=True)[0]
                if(Lx is None):
                    Lx = torch.zeros_like(x)
                Lu[:, :, i, j, k] = x
    return Lu


@torch.jit.script
def riemann_tensor(L:Tensor, Lu:Tensor)->Tensor:
    Rabcd = torch.empty((L.shape[0], L.shape[1], L.shape[1], L.shape[1], L.shape[1]), dtype=torch.float32)
    try:
        Rabcd = Rabcd.to(0)
    except:
        pass
    for a in range(L.shape[1]):
        for b in range(L.shape[1]):
            for c in range(L.shape[1]):
                for d in range(L.shape[1]):
                    f = torch.zeros((L.shape[0], L.shape[1]), dtype=torch.float32)
                    for e in range(L.shape[1]):
                        f += L[:, :, a, c, e] * L[:, :, b, e, d] - L[:, :, a, b, e] * L[:, :, c, e, d]
                    Rabcd[:, a, b, c, d] = Lu[:, b, a, c, d] - Lu[:,:, c, a, b, d] + f
    return Rabcd

@torch.jit.script
def ricci_tensor(R_upva):
    R_uv= torch.zeros((R_upva.shape[0], R_upva.shape[1], R_upva.shape[1]), dtype=torch.float32)
    try:
        R_uv = R_uv.to(0)
    except:
        pass 
    for p in range(R_upva.shape[1]):
        R_uv[:,:,:] += R_uv[:,:,p,:,p]
    return R_uv