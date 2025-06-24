import torch
from typing import Generator, Self
class BoundStochasticWalker:
    def __init__(
        self, 
        mu:torch.Tensor= 1, 
        sigma:torch.Tensor = 1,
        boundaries:tuple[float,float] = (-1,1),
        NumSteps = 1000,
        dt:float = 1e-3, 
        R:torch.distributions.Distribution = torch.distributions.Normal(0, 1),
        A0:torch.Tensor = torch.zeros((100,), dtype = torch.complex128),
        realWalk:bool = True
    )->Self:
        self.mu = mu
        self.sigma = sigma
        self.dt = torch.tensor(dt)
        self.R = R
        self.a, self.b = boundaries
        self.n = 0
        self.NumSteps = NumSteps
        self.itered_through = 0
        self.A = torch.empty((self.NumSteps, *A0.shape), dtype = A0.dtype, device= A0.device)
        self.A[0] = A0
        self.realWalk = realWalk
        self.shape = self.A.shape
        return
    def __iter__(self)->Generator[torch.Tensor, None, None]:
        return self
    
    def stepit(self)->None:
        if(self.realWalk):
            R = self.R.rsample(self.A.shape[1:]).to(self.A.dtype)
        else:
            R = torch.complex(self.R.rsample(self.A.shape[1:]), self.R.rsample(self.A.shape[1:])).to(self.A.dtype)
        self.A[self.n+1] = self.A[self.n] - self.mu*self.A[self.n]*self.dt + (self.b-self.A[self.n])*(self.a - self.A[self.n]) *self.sigma*R*self.dt.sqrt()
        return self.A[self.n+1]
    
    def __next__(self)->torch.Tensor:
        if(self.n == self.itered_through *self.NumSteps -1 and self.n != 0):
            self.T = torch.concat((self.A, torch.empty((self.NumSteps, *self.A.shape[1:]))))
        if(self.n< self.NumSteps*(self.itered_through+1)-1):
            self.A[self.n+1] = self.stepit()
            self.n+=1
            return self.A[self.n]
        else:
            self.itered_through +=1
            raise StopIteration
        
    def __getitem__(self, n)->torch.Tensor:
        return self.A[n]
    
    def __repr__(self)->str:
        return """BoundStochasticWalker(mu={mu}, sigma={sigma}, shape={shape}, itered_through={itered_through}, bounds=({a},{b}), distribution={R},\\ A={A}\\)""".format(
            mu = str(self.mu), 
            sigma = str(self.sigma), 
            shape=str(self.shape),
            itered_through = str(self.itered_through),
            R = self.R,
            A=str(self.A),
            b = str(self.b),
            a = str(self.a)
        )
    def reset(self)->None:
        self.n = 0
        self.itered_through= 0
        A0 = self.A[0].clone()
        self.A = torch.empty((self.NumSteps, *A0.shape), dtype = self.A.dtype)
        self.A[0] = A0.clone()
        return