from typing import Any
import torch
from torch.nn import Module, Linear, Sigmoid, Tanh, Softmax, Conv1d, Sequential, ReLU, LeakyReLU as LeLU
from torch import Tensor
from torch.linalg import matrix_exp as expm
from typing import Callable
from .....mathematics import einsum
from .....mathematics.algebra import ad
from .....physics.quantum.qobjs.torch_qobj import TQobj

class MagnusInspired(Module):
    def __init__(self, 
                 H:TQobj,
                 order:int = 4,
                 )->None:
        super(MagnusInspired, self).__init__()
        self.order = 4
        self.HBasis = self.gen_basis(H, order)
        self.GammaReal = Sequential(
            Linear(1, 48),
            LeLU(),
            
            Linear(48, 128),
            LeLU(),
            
            Linear(128, 256),
            LeLU(),
            
            Linear(256, 512),
            LeLU(),
            
            Linear(512, 256),
            LeLU(),
            
            Linear(256, 128),
            LeLU(),
            
            Linear(128, 48),
            LeLU(),
            
            Linear(48, self.HBasis.shape[0]),
            
        )
        self.GammaImag = Sequential(
            Linear(1, 48),
            LeLU(),
            
            Linear(48, 128),
            LeLU(),
            
            Linear(128, 256),
            LeLU(),
            
            Linear(256, 512),
            LeLU(),
            
            Linear(512, 256),
            LeLU(),
            
            Linear(256, 128),
            LeLU(),
            
            Linear(128, 48),
            LeLU(),
            
            Linear(48, self.HBasis.shape[0]),
            
        )
        return
    

    def gen_basis(self, H:TQobj, order:int = 4)->Tensor:
        L = get_combos(range(H.shape[0]), order)
        t = [H[0], H[1]]
        MX = [torch.max(H.real), torch.max(H.imag)]
        for l in L:
            m = H[l[0]]
            for j in l[1:]:
                m = ad(m, H[j],1)
            mmx = [m.real.max(), m.imag.max()]
            i = 0 
            while i<2:
                if(mmx[i]/MX[i]>1e-3):
                    t.append(m)
                if(mmx[i]>MX[i]):
                    MX[i] = mmx[i]
                i+=1
        H = TQobj(torch.stack(t), meta = H._metadata)
        return H
    
    def updateBasis(self, H)->None:
        self.HBasis = self.gen_basis(H, order=self.order)
        return 
    
    def to(self, *args:tuple, **kwargs:dict)->None:
        super(MagnusInspired, self).to(*args,**kwargs)
        self.HBasis = self.HBasis.to(*args, **kwargs)
        return
    
    def cpu(self)->None:
        super(MagnusInspired, self).cpu()
        self.HBasis.cpu()
        return

    def __call__(self, x:Tensor)->Tensor:
        R = self.GammaReal(x.real)
        I = self.GammaImag(x.real)
        Gamma = torch.complex(R, I)
        return (einsum('bij, Ab->Aij', self.HBasis, Gamma)).expm()
    
    def forward(self, x:Tensor)->Tensor:
        return self.__call__(x)


from copy import copy
def get_combos(rg, order):
    rg = sorted(list(rg))
    o = 1
    m = True
    ix = 2
    while True:
        if(m):
            m = False
            yield rg
        b = copy(rg)
        b.extend([rg[0] for i in range(o)])
        m = True
        if(o>=order):
            break
        o+=1
        
        while m:
            for r in rg:
                b[ix]=r
                yield b
            c = True
            i = 0
            while c:
                if(b[ix-i] == rg[-1]):
                    b[ix-i] = rg[0]
                    m = False
                else:
                    for j, r in enumerate(rg):
                        if(rg[j] == b[j]): 
                            b[ix-i] = rg[j+1]
                            break  
                    c=False
                    m = True
                i+=1
                if(i>ix-1):
                    c = False
        ix+=1
