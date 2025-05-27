from typing import Callable, Any, Self
import torch 
from torch import Tensor
from ..scipy_wrapper import torch_schur as schur

def parlette_recurrance(M:torch.Tensor, f:Callable[[torch.Tensor], torch.Tensor], fprime:Callable[[torch.Tensor], torch.Tensor])->tuple[Tensor, Tensor]:
    T,Q = schur(M, output = 'complex')
    F = f(T.diag()).diag_embed()
    def compute_it(i, j):
        dT = T[i,i] - T[j,j]
        if(dT == 0):
            term1 = T[i,j]*fprime(T[i,i]) 
            dT = 1
        else:
            term1 = T[i,j]*(F[i,i] - F[j,j])
        if(i+1 < j):
            k = torch.arange(i+1, j)
            comm = (F[i,k] @ T[k,j] - T[i,k] @ F[k,j])
        else:
            comm = 0
        return (term1 + comm)/dT
    idx = torch.tensor([0,1])
    i = torch.tensor(1)
    while i<T.shape[0]:
        F[idx[0], idx[1]] = compute_it(*idx)
        idx+=1
        if(idx[1] == T.shape[0]):
            i+=1
            idx[0] = 0
            idx[1] = i.clone()
    return F, Q


class SchurParletteFunction:
    def __init__(
            self, 
            fx:Callable[[Tensor, tuple[Tensor|Any], dict[Tensor|Any]], Tensor], 
            fprime:Callable[[Tensor, tuple[Tensor|Any], dict[Tensor|Any]], Tensor], 
            M:Tensor|None = None, 
            T:Tensor|None = None, 
            Q:Tensor|None = None
        )->Self:
        self.fx = fx
        self.fprime = fprime
        self.changeMatrix(M, T, Q)
        return

    def changeMatrix(
            self,
            M:Tensor|None = None, 
            T:Tensor|None = None, 
            Q:Tensor|None = None
        )->None:
        self.T,self.Q = self.getDecomp(M, T, Q)
        return 
    
    def getDecomp(self, M:Tensor|None = None,  T:Tensor|None = None, Q:Tensor|None = None)->tuple[Tensor, Tensor]:
        match (M, T, Q):
            case (_, Tensor(), Tensor()):
                T = T
                Q = Q
            case (Tensor(), None, None):
                T, Q = schur(M, output = 'complex')
            case _:
                try:
                    return self.T, self.Q
                except:
                    return None, None
        return T, Q
    
    def slow(self, *args,  M:Tensor|None = None,  T:Tensor|None = None, Q:Tensor|None = None, **kwargs)->Tensor:    
        T, Q =self.getDecomp(M, T, Q)
        F = self.fx(T.diag(), *args , **kwargs).diag_embed()
        def compute_it(i, j):
            dT = T[i,i] - T[j,j]
            if(dT == 0):
                term1 = T[i,j]*self.fprime(self.T[i,i], *args, **kwargs) 
                dT = 1
            else:
                term1 = T[i,j]*(F[i,i] - F[j,j])
            if(i+1 < j):
                k = torch.arange(i+1, j)
                comm = (F[i,k] @ T[k,j] - T[i,k] @ F[k,j])
            else:
                comm = 0
            return (term1 + comm)/dT
        i = torch.tensor(1)
        idx = torch.tensor([0,1])
        while i<T.shape[0]:
            F[idx[0], idx[1]] = compute_it(*idx)
            idx+=1
            if(idx[1] == self.T.shape[0]):
                i+=1
                idx[0] = 0
                idx[1] = i.clone()
        ret = self.Q @ F @ self.Q.conj().T
        return ret
    
    def __call__(self, *args,  M:Tensor|None = None,  T:Tensor|None = None, Q:Tensor|None = None, **kwargs)->Tensor:
        return self.Faster(*args, M=M, T=T, Q=Q, **kwargs)
    
    def Faster(self, *args,  M:Tensor|None = None,  T:Tensor|None = None, Q:Tensor|None = None, **kwargs)->Tensor:
        T, Q =self.getDecomp(M, T, Q)
        F = self.fx(T.diag(), *args , **kwargs).diag_embed()
        def getMetric(i):
            return ((AllIxs>ix[0][0]) & (AllIxs<ix[1][0])).to(T.dtype)
        AllIxs=torch.arange(T.shape[0])
        getMetric = torch.vmap(getMetric, in_dims = 1)
        F = self.fx(T.diag(),*args, **kwargs).diag_embed()
        for i in range(1,T.shape[0]):
            ix = torch.tensor([0,i])
            b = torch.arange(0,T.shape[0]-i)
            ix = (ix[:,None]+b[None,:])
            dT = T[ix[0],ix[0]]- T[ix[1], ix[1]]
            if((dT!=0).all()):
                term1 = T[ix[0], ix[1]]*(F[ix[0],ix[0]]- F[ix[1], ix[1]])
            else:
                idx = (dT != 0)
                tix = ix[...,idx]
                fix = ix[...,idx.logical_not()]
                term1 = torch.zeros_like(dT)
                term1[idx] = T[tix[0], tix[1]]*(F[tix[0],tix[0]]- F[tix[1], tix[1]])
                term1[idx.logical_not()] = T[fix[0], fix[1]]*(self.fprime(T[fix[0],fix[0]], *args,**kwargs))
                dT[idx.logical_not()] = 1
                del tix; del fix; del idx
            if(i > 1):
                k = getMetric(ix)
                comm = ((F[ix[0]]* T.T[ix[1]] - T[ix[0]]*F.T[ix[1]])*k).sum(-1)
            else:
                comm = 0
            F[ix[0],ix[1]] = (term1+ comm)/dT
        ret = Q @ F @ Q.conj().T
        return ret
