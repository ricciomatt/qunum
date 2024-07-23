
import torch

@torch.jit.script
def ventropy(p:torch.Tensor, epsi:float = 1e-8)->torch.Tensor:
    if(len(p.shape) == 2):
        Lam  = torch.linalg.eigvals(p).real
        S = torch.tensor([0.0])
        logLam = torch.log(Lam)
        ix = torch.where(torch.isnan(logLam) | torch.isinf(logLam))[0]
        logLam[ix] = 0
        S-=(Lam*logLam).sum()
    else:
        Lam  = torch.linalg.eigvals(p).real
        S = torch.zeros_like(Lam[:,0])
        for i in range(Lam.shape[1]):
            logLam = torch.log(Lam[:,i])
            ix = torch.where(torch.isnan(logLam) | torch.isinf(logLam))[0]
            logLam[ix] = 0
            S-=Lam[:,i]*logLam   
    return S