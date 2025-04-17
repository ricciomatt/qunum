import torch
import numpy as np 
from typing import Callable, Generator
from math import sqrt
from .core import getkCouplings, getvvCouplings, mu, mu_exp

from .......constants import c
from ......qobjs.dense.core import TQobj, qein
from ......qobjs import zeros_from_context
from ......qobjs.dense.instantiate import zeros_from_context, eye, eye_like
from ........mathematics.combintorix import EnumerateArgCombos, EnumUniqueUnorderedIdx
from ........mathematics.special_functions import dirac_delta
from ....ladder.jordan_winger import LazyJordanWigner
from ......qobjs.sparse_su2_dep import SU2Matrix, SU2State, mk_sparse_basis

class SparseNP:
    def __init__(
            self,
            p_states:torch.Tensor, 
            niter:int = int(1e3), dt:float = 1e-4, 
            hbar:float = 1,
            mass_values:torch.Tensor = torch.tensor([6.2, 89.7]),
            time_dependence:Callable[[torch.Tensor], torch.Tensor] =lambda t: mu(t, 1, 50_000., 1e2),
            dtype:torch.dtype = torch.complex128,
            device:torch.device = 'cpu',
            **kwargs:dict[str:torch.Tensor]
        )->None:
        self.flavor_idx:int = 1
        self.p_idx:int = 0
        self.pMag:torch.Tensor = EnumerateArgCombos(p_states.pow(2).sum(dim = 1).sqrt(), torch.arange(2))()
        self.p:torch.Tensor = EnumerateArgCombos(p_states, torch.arange(2),  ret_tensor=True)(dim = 0)
        
        self.u = time_dependence

        self.AGenerator:LazyJordanWigner = LazyJordanWigner(self.pMag.shape[0], dtype=dtype, device= device)
        self.niter:int = int(niter)
        self.m:torch.Tensor = mass_values   

        self.k = self.k_couplings()
        self.vv = self.vv_couplings()

        self.n:int = 0
        self.dt:float = dt
        self.hbar= hbar
        return
    
  

    def __iter__(self)->Generator[TQobj, None, None]:
        return self
     
    def __next__(self)->TQobj:
        if(self.n<self.niter):
            self.n+=1
            return 
        else:
            raise StopIteration
        return
    
    
    
    def k_couplings(self)->torch.Tensor:
        return getkCouplings(self.p, self.pMag[:,self.flavor_idx], m = self.m)

    def vv_couplings(self)->torch.Tensor:
        return  getvvCouplings(
            self.p.to(dtype = self.AGenerator.dtype, device = self.AGenerator.device), 
            self.pMag[:,self.flavor_idx].to(dtype = self.AGenerator.dtype, device = self.AGenerator.device), 
            EnumerateArgCombos(*(torch.arange(self.p.shape[0]) for i in range(4)))()
        )

    def __call__(self, t:torch.Tensor)->SU2Matrix:
        m =  self.u(t).to(dtype = self.Hvv.dtype, device = self.Hvv.device)
        return qein('A, ij -> Aij', torch.ones_like(m), self.Hk) - qein('A, ij -> Aij', m, self.Hvv)
    
    def createGenerator(self)->LazyJordanWigner:
        return LazyJordanWigner(self.pMag.shape[0])