import torch
from .....qobjs import TQobj, zeros_like, zeros_from_context, zeros
from .......mathematics.algebra.representations.su import get_pauli

from typing import Callable, Generator
from .... import pmns2, JordanWignerSu2
from .......mathematics.combintorix import YieldArgCombos, EnumUniqueUnorderedIdx
from .......mathematics.special_functions import dirac_delta
from .....qobjs.dense.instantiate import zeros_from_context, eye, eye_like
import numpy as np 
from torch import einsum as tein, jit, Tensor
from .....qobjs.meta.meta import QobjMeta
from typing import Tuple

def qein(indicies:str, *args:Tuple[Tensor|TQobj,...], **kwargs)->TQobj:
    ret_ten = tein(indicies, *args)
    meta = None
    for a in args:
        if(isinstance(a, TQobj)):
            meta = QobjMeta(n_particles=a._metadata.n_particles, dims=a._metadata.dims, shp=ret_ten.shape)
            break
    assert meta is not None, 'If you are looking to just do a regular einstien summation use torch.einsum(*args) not qn.einsum'
    if(isinstance(ret_ten, TQobj)):
        ret_ten.set_meta(meta)    
    else:
        ret_ten = TQobj(ret_ten, meta=meta)
    return ret_ten

class TwoPHamiltonian:
    def __init__(
            self,
            N:int, 
            p_states:torch.Tensor, 
            niter:int, dt:float = 1e-4, 
            hbar:float = 1,
            **kwargs:dict[str:torch.Tensor]
        )->None:
        assert p_states.shape[0] == 2, SyntaxError('Must be size 2 tensorfor p_states')
        
        self.N:int = N
        self.np:int = 2
        
        self.pMag:torch.Tensor = YieldArgCombos(torch.arange(2), p_states.pow(2).sum(dim = 1).sqrt())()
        self.p:torch.Tensor = YieldArgCombos(torch.arange(2), p_states, ret_tensor=True)(dim = 1)
        self.AGenerator:JordanWignerSu2 = JordanWignerSu2(self.N+self.np, assign_dims={n:2+self.np for n in range(self.N)})
        
        self.niter:int = niter
        
        self.Hk:TQobj = self.getHk()
        self.Hvv:TQobj = self.getHvv()

        self.n:int = 0
        self.dt:float = dt
        self.hbar= hbar
        self.HDiag = None
        self.U = None
        return
    
    def getHk(self)->TQobj:
        b = self.getLadder('mass')
        return qein('Aij, A', (b.dag() @ b), self.pMag[:,1].to(dtype = b.dtype))

    def getHvv(self)->TQobj:
        a = self.getLadder('flavor')
        gPQ = getvvCouplings(self.p.to(a.dtype), self.pMag.to(a.dtype),YieldArgCombos(*(torch.arange(4) for i in range(4)))())
        return qein(
            'ABCD, Aij, Bjk, Ckl, Dln-> in', 
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
                U = pmns2()[self.pMag[:,0].to(torch.int32)][:, self.pMag[:,0].to(torch.int32)]
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
        H = self.Hk+self.Hvv
        v, self.U = H.eig()
        self.HDiag:TQobj = TQobj(torch.diag_embed(v), meta = self.U._metadata)
        return self.HDiag, self.U
    
    def getU(self, t)->TQobj:
        if(self.HDiag is None):
            self.getEig()
        return self.U @ (qein('A,ij -> Aij', (-1j/self.hbar)*t , self.HDiag).expm())@ self.U.dag() 
   
    def evolveState(self, t:torch.Tensor)->TQobj:
        return self.getU(t) @ self.state


@torch.jit.script
def f(thetaP:torch.Tensor, phiP:torch.Tensor, thetaQ:torch.Tensor, phiQ:torch.Tensor)->torch.Tensor:
    return (-1j*phiP).exp()*(thetaP.cos()*thetaQ.sin())-(-1j*phiQ).exp()*(thetaQ.cos()*thetaP.sin())

@torch.jit.script
def getvvCouplings(p:torch.Tensor, u:torch.Tensor, Idx:torch.Tensor)->torch.Tensor:
    U = torch.zeros((4,4,4,4), dtype = p.dtype)
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
            U[e[0],e[1],e[2],e[3]] = f(theta[0],phi[0], theta[1], phi[1]).conj() * f(theta[2],phi[2],theta[3],phi[3])
    return U

