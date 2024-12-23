import torch
import numpy as np 
from typing import Callable, Generator
from math import sqrt
from .core import getkCouplings, getvvCouplings, mu, mu_exp
from .... import pmns2, JordanWignerSu2
from ....time_evolve import MagnusGenerator, DysonSeriesGenerator, ManySuddenApprx
from .......constants import c
from ......qobjs.dense.core import TQobj, qein
from ......qobjs import zeros_from_context
from ......qobjs.dense.instantiate import zeros_from_context, eye, eye_like
from ........mathematics.combintorix import EnumerateArgCombos, EnumUniqueUnorderedIdx
from ........mathematics.special_functions import dirac_delta
class TwoPHamiltonian:
    def __init__(
            self,
            N:int, 
            p_states:torch.Tensor, 
            niter:int, dt:float = 1e-4, 
            hbar:float = 1,
            mass_values:torch.Tensor = torch.tensor([6.2, 89.7]),
            time_dependence:Callable[[torch.Tensor], torch.Tensor] =lambda t: mu(t, 1, 50_000., 1e2),
            dtype:torch.dtype = torch.complex128,
            device:torch.device = 'cpu',
            **kwargs:dict[str:torch.Tensor]
        )->None:
        assert p_states.shape[0] == 2, SyntaxError('Must be size 2 tensorfor p_states')
        self.N:int = N
        self.np:int = 2
        self.flavor_idx = 1
        self.p_idx = 0
        self.pMag:torch.Tensor = EnumerateArgCombos(p_states.pow(2).sum(dim = 1).sqrt(), torch.arange(self.N) )()
        self.p:torch.Tensor = EnumerateArgCombos(p_states, torch.arange(self.N),  ret_tensor=True)(dim = 0)
        self.u = time_dependence

        self.AGenerator:JordanWignerSu2 = JordanWignerSu2(self.N+self.np, assign_dims={n:2+self.np for n in range(self.N)}, dtype=dtype, device= device)
        self.niter:int = niter
        self.m:torch.Tensor = mass_values   

        self.Hk:TQobj = self.getHk()
        self.Hvv:TQobj = self.getHvv()

        self.n:int = 0
        self.dt:float = dt
        self.hbar= hbar
        return
    
    def getHk(self)->TQobj:
        a = self.getLadder()
        M = self.k_couplings()
        return qein('AB, Aij, Bjk -> ik', M, a.dag(), a)

    def getHvv(self)->TQobj:
        a = self.getLadder('flavor')
        gPQ = self.vv_couplings(a.dtype, a.device)
        return qein(
                'ABCD, Aij, Bjk, Ckl, Dln -> in', 
                gPQ, 
                a.dag(), 
                a.dag(), 
                a, 
                a
            )

    def __iter__(self)->Generator[TQobj, None, None]:
        return self
    
    def getLadder(self, basis = 'flavor')->TQobj:
        assert basis in ['flavor', 'mass'], ValueError("Must be in the set {'flavor', 'mass'}")
        a = self.AGenerator()
        match basis:
            case 'flavor':
                return a
            case 'mass':
                U = pmns2()[self.pMag[:,self.flavor_idx].to(torch.int32)][:, self.pMag[:,self.flavor_idx].to(torch.int32)]
                P = torch.einsum('Ai, B->ABi', self.p, torch.ones(self.p.shape[0]))
                return qein('Ajk, AB -> Bjk',  
                            a, 
                            dirac_delta(
                               (P - torch.swapaxes(P, axis0=0,axis1=1)).pow(2).sum(dim = -1)
                            )
                            *
                            U
                        )

    def flavorToMassTransform(self)->TQobj:
        b = self.getLadder('mass')
        a = self.getLadder('flavor')
        meta = a._metadata
        p = zeros_from_context(dims = meta.dims)
        p[:, -1] = 1
        '''
            I:TQobj = eye_like(a)
            a:TQobj = torch.stack(list(I if(i == 0) else a[i-1] for i in range(5)))
            a.set_meta(meta = meta)
            b:TQobj = torch.stack(list(I if(i == 0) else b[i-1] for i in range(5)))
            b.set_meta(meta = meta)
        '''
        def filterFun(x:np.ndarray[np.int32])->bool:
            for i in np.unique(x):
                if(i != 0):
                    if((x==i).sum() > 1):
                        return False
            return True
        enum = filter(
            filterFun,
            EnumUniqueUnorderedIdx(*(
                range(a.shape[0]) for i in range(a.shape[0]-1)
                )
            )
        )
        U:TQobj = sum(
            (
                (
                    b[e].matprodcontract().dag() @ p
                ) @ (
                    a[e].matprodcontract().dag() @ p
                )[0].dag() for e in enum)
            )
        U.set_meta(
                meta = meta, inplace = False
            )
        return U
    
    def __next__(self)->TQobj:
        if(self.n<self.niter):
            self.n+=1
            return 
        else:
            raise StopIteration
        return
    
    def createState(self, n_excitations:int = 2)->TQobj:
        adag = self.AGenerator('create')
        psi = zeros_from_context(dims = adag._metadata.dims)
        psi[0,-1] = 1
        def filterFun(x:tuple|np.ndarray[np.int32])->bool:
            for i in np.unique(x):
                if((x==i).sum() > 1):
                    return False
            return True
        enum = filter(
            filterFun,
            iter(
                EnumUniqueUnorderedIdx(
                    *(
                        range(4) 
                        for i in range(
                            n_excitations
                        )
                    )
                )
            )
        )
        psi = sum((adag[e].matprodcontract() for e in enum)) @ psi
        return psi / torch.sqrt((psi.dag() @ psi))
    
    def setState(self, n_excitations:int = 2)->None:
        self.state = self.createState(n_excitations=n_excitations)
        return 
    
    def getEig(self)->tuple[TQobj, torch.Tensor]:
        H = self.Hk-self.Hvv
        v, self.U = H.eig()
        self.HDiag:TQobj = TQobj(torch.diag_embed(v), meta = self.U._metadata)
        return self.HDiag, self.U
    
    def getU(self, t:torch.Tensor)->TQobj:
        H = self(t)
        v,U = H.eig()
        return (U.dag() @ (-1j*v*(t[1]-t[0])).exp().diag_embed() @ U).cummatprod()
   
    def evolveState(self, t:torch.Tensor, psi0:TQobj)->TQobj:
        return self.getU(t) @ psi0

    def k_couplings(self)->torch.Tensor:
        return getkCouplings(self.p, self.pMag[:,self.flavor_idx], m = self.m)

    def vv_couplings(self, dtype:torch.dtype = torch.complex128, device:torch.device = 'cpu')->torch.Tensor:
        return  getvvCouplings(
            self.p.to(dtype = self.AGenerator.dtype, device = self.AGenerator.device), 
            self.pMag[:,self.flavor_idx].to(dtype = self.AGenerator.dtype, device = self.AGenerator.device), 
            EnumerateArgCombos(*(torch.arange(4) for i in range(4)))()
        )

    def __call__(self, t:torch.Tensor)->TQobj:
        m =  self.u(t).to(dtype = self.Hvv.dtype, device = self.Hvv.device)
        return qein('A, ij -> Aij', torch.ones_like(m), self.Hk) - qein('A, ij -> Aij', m, self.Hvv)

    def createGenerator(self,method:str = 'magnus', order:int = 2, dt = 1e-3)->MagnusGenerator|ManySuddenApprx:
        match method:
            case 'magnus':
                return MagnusGenerator(self, order= order, dt = dt, h_bar=self.hbar )
            case 'sudden':
                return ManySuddenApprx(self, dt = dt)
            case _:
                raise ValueError('Method must be in {"magnus", "sudden"}')

