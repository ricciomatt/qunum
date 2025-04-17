from .......mathematics.algebra import su_n_generate
from typing import Self
from torch import device as torchDevice, dtype as torchDtype, complex128 as torchComplex128, jit, Tensor, zeros, kron, einsum
from .....qobjs import TQobj, direct_prod as dprod
class sunIsingModel:
    def __init__(self,N:int = 2 ,n:int = 3, B:Tensor|None = None, p:Tensor|None = None, device:torchDevice='cpu', dtype:torchDtype = torchComplex128)->Self:
        assert self.p.shape[0] == N, ValueError('p Must be a Tensor of length {N} for a {N} particle state but found {lp}'.format(N = N, lp = p.shape[0]))
        assert B.shape[0] == n**2-1, ValueError('B must be a vector of length {n}**2 - 1 but found {alen} != {tlen}'.format(n=n, tlen=n**2-1, alen= B.shape[0]))
        self.n = n
        self.N = N
        self.dtype = dtype
        self.device = device
        self.B = B
        self.p = p
        self.generators = su_n_generate(
            self.n, 
            gen_ret_type = 'tqobj', 
            device = self.device, 
            dtype = self.dtype
        ) 
        self.genBasis()


    def genBasis(self:Self,)->None:
        self.Hk = self.kineticTerm()
        return 
    
    def kineticTerm(self)->TQobj:
        Mat = einsum('A, Aij', self.B, self.generators[1:]) 
        return sum([
            dprod(
                *[
                    Mat/p
                    if m == n else 
                    self.generators[0] 
                    for m in range(self.N)
                ]
            )
            for (n,p) in enumerate(self.p)
        ])

    def vvTerm(self:Self,)->TQobj:
        sum([
            dprod(
                *[
                    self.generators[0]/p
                    if m == n else 
                    self.generators[0] 
                    for m in range(self.N)
                ]
            )
            for (n,p) in enumerate(self.p)
        ])
        return
