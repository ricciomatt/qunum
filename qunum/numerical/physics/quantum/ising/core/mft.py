import torch
from ...qobjs import TQobj
from .....mathematics.algebra.sun.structure import SUConnection as SUAlgebra, su_n_generate
from .....mathematics.stochastic.bound import BoundStochasticWalker
from typing import Self
from ...operators.dense.nuetrino import pmns2, pmns3
class sunIsing:
    def __init__(
        self, n_alg:int, 
        p:torch.Tensor|None = None, 
        dm2:float=1, Dm2:float=3, 
        dt = 1e-3, 
        rho0:TQobj|None = None, 
        Rv:float = 50e6, r0:float = 50e6,
        c:float = 3e8, u0:float = 5.,
        NumSteps:int = 1000,
        use_const_mu:bool = False,
        metropolisTemperature:torch.Tensor = torch.tensor(1.),
        metroplisDiffusion:torch.Tensor = torch.tensor(1),
        useMetropolis:bool = False,
        **BoundWalkerParams:dict
    )->Self:
        self.sun = SUAlgebra(n_alg)
        self.B = torch.zeros((self.sun.n**2-1,), dtype = torch.complex128)
        self.p = p
        self.Rv = torch.tensor(Rv)
        self.r0 = torch.tensor(r0)
        self.c = torch.tensor(c)
        self.u0 = torch.tensor(u0)
        T = su_n_generate(self.sun.n, include_identity = 0, ret_type='tqobj')
        match (rho0, n_alg):
            case (None,3):
                psi0 = TQobj(torch.zeros((self.sun.n,1),dtype = torch.complex128))
                psi0[0] = 1                
                psi0 = pmns3() @ psi0
                rho = psi0 @ psi0.dag()
                self.B[2] = dm2/4
                self.B[8] = Dm2/4
            case (None,2):
                psi0 = TQobj(torch.zeros((self.sun.n,1),dtype = torch.complex128))
                psi0[0] = 1                
                psi0 = pmns2() @ psi0
                rho = psi0 @ psi0.dag()
                self.B[2] = dm2/4
            case _:
                assert rho0.shape[-1] == self.sun.n, ValueError('rho0 must be the same dim as the algebra')
                pass
        Texp0 = (rho @ T).Tr()
        self.rho = rho
        Texp0 = torch.stack([Texp0.clone() for i in range(self.p.shape[0])])
        self.NumSteps = NumSteps
        self.T = torch.empty((self.NumSteps, *Texp0.shape), dtype = Texp0.dtype, device = Texp0.device)
        self.T[0] = Texp0.clone()
        self.n = 0
        self.itered_through = 0 
        self.dt = torch.tensor(dt)
        self.use_const_mu = use_const_mu
        self.t0 = 0
        
        BoundWalkerParams['NumSteps'] = NumSteps
        self.At = BoundStochasticWalker(A0=torch.zeros((self.p.shape[0],self.p.shape[0]),dtype = torch.complex128), dt = dt,**BoundWalkerParams) 
        self.Ec = self.computeMeanFieldE(self.T[0])
        self.E0 = torch.einsum('AB, Aa, Ba->', 1-torch.eye(self.p.shape[0], dtype = self.T.dtype, device = self.T.device), self.T[0], self.T[0])/(2*self.p.shape[0])
        self.Ru = torch.distributions.Uniform(0,1)
        self.metropolisTemperature = metropolisTemperature
        self.metropolisDiffusion = metroplisDiffusion
        self.useMetropolis:bool = useMetropolis
        return 
    
    def u(self, t:torch.Tensor|float)->torch.Tensor:
        if(self.use_const_mu):
            return self.u0
        else:
            return self.u0*(
                (
                    1 - torch.sqrt(1- (self.Rv/(self.r0 + self.c*t))**2)
                )
            ).pow(2)
    
    def meanFieldDynamics(self)->torch.Tensor:
        Hk = self.kineticStep()
        Hv = self.potentialStep()
        return Hk + Hv
    
    #Stepping Functions
    def DynamicalStep(self)->torch.Tensor:
        dHMeanField= self.meanFieldDynamics()
        dHCorrection = self.correlationStep()
        return dHMeanField + dHCorrection

    def kineticStep(self)->torch.Tensor:
        return torch.einsum(
            'Ab,abc, Ac ->Aa', 
            torch.einsum(
                'A, b->Ab', 
                self.p.pow(-1), 
                self.B
            ),
            self.sun.f, 
            self.T[self.n]
        )*self.dt
    
    def potentialStep(self)->torch.Tensor:
        return (
            (self.u(self.t0+ self.n*self.dt)/self.p.shape[0]) * 
            torch.einsum(
                'ABa, AB -> Aa', 
                torch.einsum(
                    'abc, Ac, Bb-> ABa', self.sun.f,  self.T[self.n], self.T[self.n]
                ), 
                1-torch.eye(self.p.shape[0], dtype = self.T.dtype, device = self.T.device)
            ) * self.dt) 
    
    def correlationStep(self)->torch.Tensor:
        s = self.var(self.T[self.n]).sqrt()
        u = self.u(self.dt*self.n)
        return (
            (u/self.p.shape[0])*torch.einsum(
                'AB, Aa, Bq, abc-> Ac',
                (1-torch.eye(self.p.shape[0], dtype = self.T.dtype, device = self.T.device))*self.At[self.n], 
                s, 
                s, 
                self.sun.f
            ) * 
            self.dt
        )
    #Compute Quantities
    def var(self, T:int|None = None)->torch.Tensor:
        match T:
            case None:
                return (2/self.sun.n + 1/2 * torch.einsum('ab, Ab-> Aa', self.sun.d[torch.arange(self.sun.n**2 - 1), torch.arange(self.sun.n**2 - 1)], self.T[self.n]) - self.T[self.n]**2)
            case torch.Tensor():
                return (2/self.sun.n + 1/2 * torch.einsum('ab, Ab-> Aa', self.sun.d[torch.arange(self.sun.n**2 - 1), torch.arange(self.sun.n**2 - 1)], T) - T**2)
    
    #Compute The Energy
    def computeE(self, n:int|None = None, T:torch.Tensor|None = None, A:torch.Tensor|None = None, s:torch.Tensor|None = None, t:torch.Tensor|float|None = None)->torch.Tensor:
        match (n, T, A, s, t):
            case (int(), _, _, _,_):
                T = self.T[n].clone()
                A = self.At[n].clone()
                s = self.var(T=T).sqrt()
                t = n*self.dt
            case (None, torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()|float()):
                pass
            case _:
                T = self.T[self.n].clone()
                A = self.At[self.n].clone()
                t = self.n*self.dt
                pass
        Hmf = self.computeMeanFieldE(n=n, T=T, t=t)
        Hr = self.computeCorrelationEstimator(n = n, T = T, A = A, t = t)
        return Hmf + Hr

    #Compute The Mean Field Energy 
    def computeMeanFieldE(self, n:int|None = None, T:torch.Tensor|None = None, t:torch.Tensor|None = None):
        match (n, T, t):
            case (int(), _,_,_):
                T = self.T[n].clone()
                A = self.At[n].clone()
                t = n*self.dt
            case (None, torch.Tensor(), torch.Tensor()|float()):
                pass
            case _:
                T = self.T[self.n].clone()
                A = self.At[self.n].clone()
                t = self.n*self.dt
                pass
        return self.computeKintecExp(T = T)+self.computePotentialExp(T=T, t= t)
    
    #Computing the Correlation Estimator.
    def computeCorrelationEstimator(self, n:int|None = None, s:torch.Tensor|None = None, A:torch.Tensor|None = None, t:torch.Tensor|None = None):
        match (n, A, s, t):
            case (int(), _, _, _,_):
                A = self.At[n].clone()
                s = self.var(T=T).sqrt()
                t = n*self.dt
            case (None, torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()|float()):
                pass
            case _:
                s = self.var(T= self.T[self.n].clone()).sqrt()
                A = self.At[self.n].clone()
                t = self.n*self.dt
                pass
        return torch.einsum('AB, Aa, Ba->', A*(1-torch.eye((A.shape[0]), dtype =A.dtype )), s,s)
        
    def computeKintecExp(self,n:int|None = None, T:torch.Tensor|None = None)->torch.Tensor:
        match (n,T):
            case (int(), _):
                T = self.T[n].clone()
            case (None, torch.Tensor()):
                pass
            case _:
                T = self.T[self.n].clone()
        return torch.einsum('Aa, Aa ->', T, 
                     torch.einsum(
                        'A, b->Ab', 
                        self.p.pow(-1), 
                        self.B
                    )
                )
    
    def computePotentialExp(self, n:int|None = None, T:torch.Tensor|None = None , t:torch.Tensor|float|None = None)->torch.Tensor:
        match (n, T, t):
            case (int(), _,_,_):
                T = self.T[n].clone()
                A = self.At[n].clone()
                t = n*self.dt
            case (None, torch.Tensor(), torch.Tensor()|float()):
                pass
            case _:
                T = self.T[self.n].clone()
                A = self.At[self.n].clone()
                t = self.n*self.dt
                pass
        U = torch.einsum('Aa, Ba, AB -> ', T, T, 1-torch.eye(self.p.shape[0], dtype = T.dtype, device = T.device))
        u = self.u(t)
        return u/(2*self.p.shape[0])*(U)
    
    def computeVarAll(self)->torch.Tensor:
        return (2/self.sun.n + 1/2 * torch.einsum('ab, ABb-> ABa', self.sun.d[torch.arange(self.sun.n**2 - 1), torch.arange(self.sun.n**2 - 1)],self.T) - self.T**2)
    
    def monteCarlo(self)->torch.Tensor:
        self.T[self.n]= self.T[self.n].clone() + self.DynamicalStep()
        Hmf = self.computeMeanFieldE()
        s = self.var(T = self.T[self.n+1]).sqrt()
        P = []
        while True:
            Atpdt = self.At.stepit()
            E = Hmf + self.computeCorrelationEstimator(s = s, A = Atpdt, t = (self.n+1)*self.dt)
            PAccept = self.metropolisCriteron(E)
            if(self.metropolisTemperature*self.Ru.sample()<PAccept):
                print('Accept')
                self.At.n+=1
                self.Ec = E 
                print(P)
                return 
            return
       
    def metropolisCriteron(self, E:torch.Tensor)->torch.Tensor:
        return torch.exp(
            -(
                (
                    (
                        E - self.Ec
                    )
                    - 
                    (
                        self.u(self.dt*(self.n+1))-self.u(self.dt*self.n)
                    )*self.E0
                ).pow(2)
                /(
                    2*(self.metropolisDiffusion**2)*(self.dt**2)
                ) 
            )
        ).real
    
    def __iter__(self)->Self:
        return self
    
    def __next__(self)->tuple[torch.Tensor,torch.Tensor]:
        if(self.n == self.itered_through *self.NumSteps -1 and self.n != 0):
            self.T = torch.concat((self.T, torch.empty((self.NumSteps, *self.T.shape[1:]))))
        if(self.NumSteps*(self.itered_through+1)-1>self.n):
            self.T[self.n+1] = self.T[self.n] + self.meanFieldDynamics()
            #self.monteCarlo()
            self.n+=1
            return self.T[self.n]
        else:
            self.itered_through += 1
            raise StopIteration

    def step(self)->torch.Tensor:
        return next(self)
    
    def change_dt(self,dt:float)->torch.Tensor:
        self.t0 = self.dt*self.n
        self.dt = dt

