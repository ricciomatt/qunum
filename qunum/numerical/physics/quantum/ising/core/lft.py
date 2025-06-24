from typing import Generator, Self, Callable
from warnings import warn
from polars import LazyFrame, DataFrame, col, from_torch, from_numpy, concat_arr, concat_list, when, Array, Int32
from torch import Tensor, tensor, sqrt, rand, complex128, float64, float32, empty, zeros
import torch
#local imports
from ...qobjs import TQobj
from ...operators.dense.nuetrino import pmns2, pmns3
from .....mathematics.stochastic.bound import BoundStochasticWalker
from .....mathematics.combintorix import EnumerateArgCombos, EnumUniqueUnorderedIdx
from .....mathematics.algebra.sun.structure import SUConnection as SUAlgebra, su_n_generate
from .....mathematics.tensors import LazyTensor
from .....mathematics.numerics import AdaptiveLinspace

def uf(t:Tensor, u0:float=5, Rv:float = 50e6, r0 = 50e6, c = 3e8)->Tensor:
    match t:
        case Tensor():
            return u0*(
                (
                    1 - sqrt(1- (Rv/(r0 + c*t))**2)
                )
            ).pow(2)
        case _:  
            return u0*(
                (
                    1 - sqrt(1- (Rv/(r0 + c*tensor(t)))**2)
                )
            ).pow(2)
class HotIsingMcMc:
    def __init__(
        self, 
        N:int, 
        n:int, 
        p_states:Tensor|None = None,
        rho:TQobj|None = None, 
        Nsteps:int|None = int(1e2), dt:float = 1e-1,
        B:Callable[[Tensor], Tensor]|Tensor|None = None,
        u:Callable[[Tensor],Tensor]|None = None,
        method_:str|None =None,
        adaptive_time_step:bool = False,
        stoch_var:float = 1,
        u0:float = 1e2,
    )->Self:
        
        if(method_ is None): self.method_ = 'mft'
        else: self.method_ = method_
        self.n = 0
        self.sun:SUAlgebra = SUAlgebra(n)
        self.rho0:TQobj = self.__get_state(rho)
        T0:Tensor = self.__initialize_expvals().clone().real
        self.Nsteps:int = Nsteps
        self.T:Tensor = empty((self.Nsteps, N, n**2 - 1))
        self.T[0] = T0
        
        self.sun.f = self.sun.f.to(self.T.dtype)
        if(p_states is not None):self.p:Tensor = p_states
        else: self.p:Tensor = torch.rand(N)*torch.randint(1,100,(N,))

        

        self.LambdaIX:Tensor = EnumerateArgCombos(range(self.sun.n**2-1), range(self.sun.n**2 -1)).__tensor__()
        self.ParticleIX:Tensor = EnumUniqueUnorderedIdx(range(N), range(N)).__tensor__()
        self.ParticleIX:Tensor = self.ParticleIX[self.ParticleIX[:,0] != self.ParticleIX[:,1]]
        self.GammaMap:LazyFrame = from_torch(
            EnumerateArgCombos(self.ParticleIX, self.LambdaIX[self.LambdaIX[:,0]!=self.LambdaIX[:,1]]).__tensor__().flatten(-2), 
            schema={"A":Int32,"B":Int32,"a":Int32,"b":Int32,}
        ).with_row_index('At').select(
            concat_list('A','B').alias('Pix'),
            'At', 
            concat_arr('A','B','a','b').alias('AB,ab'),
        ).explode(
            'Pix'
        ).with_columns(
            when(
                col('AB,ab').arr.get(0).__eq__(col('Pix'))
            ).then(
                col('AB,ab')
            ).otherwise(
                concat_arr([col('AB,ab').arr.get(1),col('AB,ab').arr.get(0), col('AB,ab').arr.get(3),col('AB,ab').arr.get(2)])
            )
        ).group_by(
            'Pix'
        ).agg('At', 'AB,ab').with_columns(
            col('At').cast(
                Array(Int32,(self.T.shape[1]-1)*(self.sun.n**2-1)*(self.sun.n**2-2),)
            ),
            col('AB,ab').cast(
                Array(Int32, ((self.T.shape[1]-1)*(self.sun.n**2-1)*(self.sun.n**2-2), 4))
            )
        ).sort('Pix').lazy()
        self.ThetaMap:Tensor = EnumerateArgCombos(self.ParticleIX, self.LambdaIX[self.LambdaIX[:,0]==self.LambdaIX[:,1]]).__tensor__()
        self.B, self.u = self.__initialize_timeDependence(B,u, u0)
        self.dt:Tensor = AdaptiveLinspace(adaptive_function=self.u if adaptive_time_step else None, dt = dt,Nsteps = self.Nsteps)
        self.E0 = self.__initial_Energy()
        self.t0 = 0
        self.Rt = BoundStochasticWalker(
            A0 = zeros((self.T.shape[1]*(self.T.shape[1]-1)*(self.sun.n**2-1)*(self.sun.n**2-2))//2, dtype = float64), 
            sigma = stoch_var, NumSteps=self.Nsteps
        )
        self.new_trajectory()
        return 
    
    #Iteration
    def __iter__(self)->Generator[Tensor, None, None]:
        return self   
    
    def __next__(self)->Tensor:
        if(self.n < self.T.shape[0]-1):
            self.__iter_Solve()
            self.n+=1
            return self.T[self.n]
        else:
            raise StopIteration
    
    def __iter_Solve(self):
        dt, t = self.__get_t_and_dt(self.n)
        match self.method_.lower():
            case 'kinetic':
                self.T[self.n+1] = self.T[self.n] + self.__compute_kineticStep(t = t, n = self.n, T = self.T[self.n])*dt
            case 'meanfield':
                self.T[self.n+1] = self.T[self.n] +  self.__compute_meanFieldStep(t = t, n = self.n, T = self.T[self.n])*dt
            case 'mft':
                self.T[self.n+1] = self.T[self.n] + self.__compute_meanFieldStep(t = t, n = self.n, T = self.T[self.n])*dt
            case 'mftrk':
                self.T[self.n+1] = self.T[self.n] + self.__computeRKMeanField(t,dt=dt,n=self.n, T=self.T[self.n])
            case 'mft+stoch_raw':
                self.T[self.n+1] = self.T[self.n] + self.__compute_meanFieldStep(t = t, n = self.n, T = self.T[self.n])*dt + self.__compute_stochasticCorrectionStep(t = t, n = self.n, T = self.T[self.n])*dt
            case 'lft':
                self.T[self.n+1] = self.T[self.n] + self.__compute_meanFieldStep(n = self.n, t = t, dt = dt)*dt + self.__compute_stochasticCorrectionStep(n = self.n, t = t, dt = dt)*dt
            case 'mcmc':
                pass
            case 'hmc':
                pass
            case _:
                warn('Assuming MeanField, unkonwn method passed "{me}" only allows ["meanfield", "lft", "mcmc"]'.format(me=self.method_))
        self.t0 = t
        return
    
    #MeanField Steps
    def __compute_kineticStep(self, t:Tensor, n:int|None = None, T:Tensor|None = None)->Tensor:
        T = self.__getT(n=n,T=T)
        aix = self.LambdaIX[self.LambdaIX[:,0] != self.LambdaIX[:,1]]
        return (
                (
                    (T[:,aix[:,1]] * self.B(t)[:,aix[:,0]]) @ self.sun.f[:, aix[:,0], aix[:,1]].T
                )
            )
    def __compute_meanInteractionStep(self, t:Tensor, n:int|None = None, T:Tensor|None = None)->Tensor:
        T = self.__getT(n=n,T=T)
        ABabIx = self.GammaMap.collect()['AB,ab'].to_numpy()
        return (
                self.u(t) * ((T[ABabIx[...,0], ABabIx[...,2]]*T[ABabIx[...,1], ABabIx[...,3]]) * self.sun.f[:,ABabIx[...,3], ABabIx[...,2]]).sum(-1).T
        )
    
    def __compute_meanFieldStep(self, t:Tensor, n:int|None = None, T:Tensor|None = None)->Tensor:
        return (self.__compute_kineticStep(t=t, n = n, T = T) + self.__compute_meanInteractionStep(t = t, n = n, T = T))
    
    def __computeRKMeanField(self, t:Tensor, dt:Tensor, n:int|None = None, T:Tensor|None = None, ):
        self.__getT(n = n, T = T)
        k1 = self.__compute_meanFieldStep(t=t, n=n, T=T)
        k2 = self.__compute_meanFieldStep(t=t+dt/2, T = T + k1*dt/2)
        k3 = self.__compute_meanFieldStep(t=t+dt/2, T = T + k2*dt/2)
        k4 = self.__compute_meanFieldStep(t=t+dt, T = T + k3*dt)
        return dt/6*(k1+2*k2+2*k3+k4)
    
    
    #Correction Functions
    def __compute_stochasticCorrectionStep(self, t:Tensor, n:int|None = None, T:Tensor|None = None)->Tensor:
        ix = self.GammaMap.collect()['At'].to_numpy()
        fix = self.GammaMap.collect()['AB,ab'].to_numpy()
        return self.u(t)*(self.At[n, ix]*self.sun.f[:,fix[...,3],fix[...,2]]).sum(-1).T 
    
    # Standard Get Fucntions
    def __getT(self,n:int|None = None, T:Tensor|None = None)->Tensor:
        match (n,T):
            case _, Tensor:
                return T
            case int, None:
                return self.T[n]
            case None, None:
                return self.T[self.n]
    
    def __get_t_and_dt(self, n:int)->tuple[Tensor,Tensor]:
        return self.dt[n]
    
    #Energy            
    def __compute_CorrectionH(self, n:int|None = None)->Tensor:
        print('core')
        match n:
            case int():
                return self.u(self.dt.t[n])*self.Dt[n].sum()
            case _:
                return self.u(self.dt.t[:-1])*self.Dt.sum(-1)
             
    def __compute_MeanH(self, n:int|None = None):
        match n:
            case int():
                return (
                    (
                        self.T[n] * self.B(self.dt.t[n])
                    ).sum(dim = list(range(1,len(self.T.shape)))) + 
                    (
                        self.u(n*self.dt.t[n])/2/self.T.shape[1]
                    ) * (
                        (
                            self.T[n, self.ThetaMap[:,0,0], self.ThetaMap[:,1,0]]* self.T[n, self.ThetaMap[:,0,1], self.ThetaMap[:,1,1] ]
                        ).sum(dim = 1)
                
                    )
                )
            case _:
                return (
                    (
                        self.T * self.B(self.dt.t[0])[None, :]
                    ).sum(dim = list(range(1,len(self.T.shape)))) + 
                    (
                        self.u(self.dt.t[:-1])
                    ) * (
                        (
                            self.T[:, self.ThetaMap[:,0,0], self.ThetaMap[:,1,0]]* self.T[:, self.ThetaMap[:,0,1], self.ThetaMap[:,1,1] ]
                        ).sum(dim = 1)
                
                    )
                )

    def __compute_Energy(self, n:int|None = None, method_:str|None = None)->Tensor:
        if(method_ is None):
            method_ = self.method_
        match (n, method_.lower()):
            case int(), 'mft'|'mftrk':
                return self.__compute_MeanH(n = n)
            
            case int(), _:
                return self.__computeMeanH(n = n) + self.__compute_CorrectionH(n = n)
            case _, 'mft'|'mftrk':
                return self.__compute_MeanH()
            case _, _:
                return self.__compute_MeanH() + self.__compute_CorrectionH()

    #initialize

    def __initial_Energy(self)->Tensor:
        return (
            self.T[0]*self.B(0)
        ).sum() + (
            self.u(0)
        )
       
    def __initialize_expvals(self)->Tensor:
        gell_mann = su_n_generate(self.sun.n, ret_type='TQobj', oplusu1=False)
        return (self.rho0 @ gell_mann).Tr()
        
    def __get_state(self, rho:TQobj|None = None)->None:
        match (self.sun.n, rho):
            case 2, None:
                rho = (pmns2() @ TQobj(tensor([[1+0j],[0+0j]], dtype= complex128))).to_density()
            case 3, None:
                rho = (pmns3() @ TQobj(tensor([[1+0j],[0+0j], [0+0j]], dtype= complex128))).to_density()
            case _,None:
                raise ValueError('Cannot infer rho0 except for n = 2 and n = 3 level case')
            case _, TQobj():
                assert self.sun.n == rho.shape[-1]  and self.sun.n == rho.shape[-2], ValueError('Density matrix provided does not have proper shape expected ({n},{n}) but found {shape}'.format(n = self.sun.n, shape = rho.shape))
        return rho
    
    def __initialize_timeDependence(
        self, 
        B:Callable[[Tensor], Tensor]|Tensor|None = None, 
        u:Callable[[Tensor], Tensor]|None = None,
        u0:float|None = None,
        dm2:float=1, Dm2:float=3
    )->None:
        match (B, self.sun.n):
            case (None,2):
                tB = zeros((self.sun.n**2-1), dtype = self.T.dtype)
                tB[2] = dm2/4
                tB = tB[None,:]/ self.p[:,None]
                tB = tB.to(self.T.dtype)
                B = LazyTensor(lambda t: tB,dtype=self.T.dtype)
            case (None, 3):
                tB = zeros((self.sun.n**2-1), dtype = self.T.dtype)
                tB[2] = dm2/4
                tB[-1] = Dm2/4
                tB = tB[None,:]/ self.p[:,None]
                
                tB = tB.to(self.T.dtype)
                B = LazyTensor(lambda t: tB,dtype=self.T.dtype)
            case (None, _):
                warn('Chosing to initialze a random B')
                tB =rand((self.T.shape[0], self.sun.n**2 -1 ), self.T.dtype)
                
                tB = tB.to(self.T.dtype)
                B = LazyTensor(lambda t: tB,dtype=self.T.dtype)
            case (_,_):
                B = LazyTensor(B,dtype = self.T.dtype)
        match (u, self.sun.n):
            case (None, _):
                if(u0 is None): u0 = 0
                u = LazyTensor(lambda t: uf(t, u0=u0))/2/self.T.shape[1]
            case (_,_):
                u = LazyTensor(u)
        
        return B, u
    
    #Public Functions
    def get_Step(self, n:int, tp:str = 'kinetic'):
        match tp:
            case 'kinetic':
                return self.__compute_kineticStep(n = n, dt =self.dt.dt[n], t= self.dt.t[n], T= self.T[n])
            case 'V':
                return self.__compute_meanInteractionStep(n=n,dt=self.dt.dt[n], t= self.dt.t[n])
   
    def Energy(self)->Tensor:
        return self.__compute_Energy()
    
    def fit(self)->None:
        self.new_trajectory()
        self.n = 0 
        for s in self:
            pass
        return 
    
    def new_trajectory(self,)->None:
        for r in self.Rt:
            pass
        self.At = self.Rt.A.clone()
        self.Rt.reset()
        for r in self.Rt:
            pass
        self.Dt = self.Rt.A.clone()
        self.Rt.reset()
        return
