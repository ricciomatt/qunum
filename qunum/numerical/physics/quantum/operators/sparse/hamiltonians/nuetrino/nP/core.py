
import torch

@torch.jit.script
def f(thetaP:torch.Tensor, phiP:torch.Tensor, thetaQ:torch.Tensor, phiQ:torch.Tensor)->torch.Tensor:
    return (-1j*phiP).exp()*(thetaP.cos()*thetaQ.sin())-(-1j*phiQ).exp()*(thetaQ.cos()*thetaP.sin())

@torch.jit.script
def getvvCouplings(p:torch.Tensor, u:torch.Tensor, Idx:torch.Tensor)->torch.Tensor:
    U = torch.zeros((p.shape[0],p.shape[0],p.shape[0],p.shape[0]), dtype = p.dtype)
    for e in Idx:
        if(
            ( 
               ((p[e[0]] + p[e[1]]) - (p[e[2]] + p[e[3]])).pow(2).sum() == 0
            )  
            and 
            (
                (
                    (u[e[0]] == u[e[2]]) 
                    and 
                    (u[e[1]]== u[e[3]])
                ) or 
                (
                    (u[e[0]]==u[e[3]]) 
                    and 
                    (u[e[1]]==u[e[2]]))
            )
        ):
            mag = p[e].pow(2).sum(dim=1, keepdim=True)
            pHat = p[e]/mag
            phi = pHat[:,2].cos()
            theta = (pHat[:,0]/(phi.sin())).sin()
            U[e[0],e[1],e[2],e[3]] = f(theta[0], phi[0], theta[1], phi[1]).conj() * f(theta[2], phi[2], theta[3], phi[3])
    return U

@torch.jit.script
def getkCouplings(p:torch.Tensor, flavor:torch.Tensor, theta:torch.Tensor= torch.tensor(torch.pi/5), m=torch.arange(2)):
    A = torch.zeros(flavor.shape[0], flavor.shape[0], dtype = torch.complex128)
    sm = m.sum()
    dm = (m[1] - m[0])
    for i, v in enumerate(flavor):
        for j, u in enumerate(flavor):
            if((p[i]- p[j]).pow(2).sum() == 0):
                q = (p[i] @ p[i]).sqrt()
                if(u == v):
                    A[i,j] += q + (sm.pow(2) - ((-1)**u)*(dm.pow(2))*torch.cos(2*theta))/q
                else:
                    A[i,j] += (dm.pow(2)*torch.sin(2*theta))/q    
    return A

def mu(t:torch.Tensor, c:float, Rv:torch.Tensor, mu0:float, r0:None|float = None)->torch.Tensor:
    if(r0 is None):
        r0 = Rv
    return mu0*(
        1-
        (
                1-
            (
                (
                    Rv
                )/(
                    r0 + c*t
                )
            ).pow(2)
        ).sqrt()
    ).pow(
        2
    )


def mu_exp(t:torch.Tensor, alpha:float, mu0:float)->torch.Tensor:
    return -(alpha*t).exp()*mu0