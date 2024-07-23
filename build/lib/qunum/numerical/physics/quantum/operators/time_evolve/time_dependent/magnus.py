from ....qobjs import TQobj
import torch 
from typing import Callable, Iterable, Iterator, Any, Generator
from ......mathematics.numerics.integrators.newton import NewtonCoates, integrate_newton_coates_do as newton
from ......mathematics.algebra import ad
from scipy.special import bernoulli
from IPython.display import display as disp, Markdown as md 
import numpy as np

class MagnusGenerator:
    def __init__(
            self, 
            Hamiltonian:Callable[[torch.Tensor], TQobj], 
            order:int = 2, 
            dt:float|torch.Tensor = float(1e-3), 
            num_pts:int = int(1e3), 
            niter_eval:int = int(50), 
            Integrator:NewtonCoates = NewtonCoates(order = 2, dtype=torch.complex128),
            h_bar:float = 1.0,
            **kwargs:dict[str:Any]
        )->object:
        assert isinstance(Integrator, NewtonCoates), 'Must be instance of NewtonCoates Integrator'
        assert callable(Hamiltonian), 'Hamiltonian Must be callable and take in only a time argument'
        self.Hamiltonian = Hamiltonian
        self.set_order(int(order))
        self.dt = float(dt)
        self.num_pts = int(num_pts)
        self.InterpolationCoefficents = Integrator.L
        self.niter_eval = int(niter_eval)
        self.n = 0
        self.n0 = 0
        self.hbar = float(h_bar)
        return

    def set_order(self, order:int)->None:
        self.order = int(order)
        self.BrenouliCoefficents = torch.tensor(
            bernoulli(int(order)),
            dtype=torch.complex128
        )
        return
    
    def evolve(self, a:float|torch.Tensor, b:float|torch.Tensor, num_pts:int|None = None)->TQobj:
        if(num_pts is None):
            num_pts = self.num_pts
        return self(torch.linspace(a, b, num_pts))

    def __call__(self, t:torch.Tensor)->TQobj:
        O = self.getOmega(t)
        return (sum(O)).expm()
    
    def getOmega(self, t)->Generator[TQobj, None, None]:
        H = self.Hamiltonian(t)
        O = expansion(H = H.to_tensor(), Bk=self.BrenouliCoefficents, order=self.order, dx = (t[1]-t[0]), L = self.InterpolationCoefficents, hbar=self.hbar)
        O = [TQobj(O[o], meta=H._metadata) for o in range(self.order)]
        del H
        return O 
        
    
    def __getitem__(self, ix:tuple[int,int]|torch.Tensor|Iterable[int]|int)->TQobj:
        if(isinstance(ix, tuple)):
            return self.evolve(ix[0], ix[1], self.num_pts)
        if(isinstance(ix, torch.Tensor)):
            assert (torch.Tensor.dtype not in [torch.int16, torch.int32, torch.int64, torch.int8]), 'Must be Integer Index'
            return self(ix*self.dt)
        if(isinstance(ix, list) or isinstance(ix, np.ndarray)):
            ix = np.array(ix).astype(np.int64)
            return self(torch.from_numpy(ix)*self.dt)
        elif(isinstance(ix,int)):
            return self(torch.tensor([ix])*self.dt)
        else:
            raise ValueError('Could Not resolve Item')
    
    def __iter__(self)->Iterator:
        return self
    
    def __next__(self)->TQobj:
        if(self.n<self.num_pts):
            t = torch.linspace(self.n*self.dt - self.dt/2, (self.n+1/2)*self.dt, self.niter_eval)
            self.n+=1
            return self(t)
        else:
            self.n0+=self.n
            self.n = 0
            raise StopIteration
    
    def reset_iter(self)->None:
        self.n0 = 0
        self.n = 0
        return
    
    def __repr__(self)->str:
        return f"{str(self.Hamiltonian)} dt={self.dt}, order={self.order},"


@torch.jit.script
def comm(A:torch.Tensor,B:torch.Tensor):
    return A @ B - B @ A


@torch.jit.script
def expansion(H:torch.Tensor, 
              Bk:torch.Tensor, 
              order:int,
              dx:float,
              L:torch.Tensor,
              hbar:float,
            )->torch.Tensor:
    H *= torch.complex(torch.tensor(0.0), torch.tensor(-1/hbar)).to(dtype = H.dtype, device = H.device)
    Omega = torch.zeros((order, H.shape[0], H.shape[1], H.shape[1]), dtype = H.dtype, device=H.device)
    Omega[0] = newton(H.clone(), L).cumsum(dim=0)*dx
    S = torch.zeros((order, order, H.shape[0], H.shape[1], H.shape[1]), dtype = H.dtype, device=H.device)
    if(order >= 2):
        for k in range(2, order+1):
            n = k-1
            for i in range(1, k):
                j = i-1
                if(i == 1):
                    t = H.clone()
                    S[n, j] = comm(Omega[n-1], t)   
                elif(i == n):
                    S[n, j] = ad(Omega[0], H.clone(), j)
                else:
                    for m in range(1, n-j):
                        S[n, j] += comm(Omega[n-m], S[n-m,j-1])
                
                Omega[n] +=  Bk[i]/torch.math.factorial(i)*newton(S[n,j], L).cumsum(dim = 0)*dx
    return Omega

#@torch.jit.script
def get_expansion_tex( 
              order:int,
              disp_:bool = False,
              print_:bool = True,
            )->tuple[dict[str:str], dict[tuple[int,int]:str]]:
    st = {'\\Omega_{0}(t)':'0', "\\Omega_{1}(t)":" \\int_{t_0}^{t} dt' H(t')"}
    S = {}
    if(order >= 2):
        for k in range(2, order+1):
            n = k-1
            temp = []
            for i in range(1, k):
                j = i-1
                if(i == 1):
                    temp.append('\\frac{B_{'+str(i)+'}}{{'+str(i)+"}!} [\\Omega_{"+str(n)+"}(t'), (-iH(t'))]")
                    S[n,j] = '[\\Omega_{'+str(n)+'}, (-iH)]'
                elif(i == n):
                    temp.append('\\frac{B_{'+str(i)+'}}{{'+str(i)+'}!} ad^{'+str(j+1)+"}_{\\Omega_{1}(t')} (-iH(t'))")
                    S[(n,j)] = 'ad^{'+str(j+1)+"}_{\\Omega_{1}(t')} (-iH(t'))"
                else:
                    t = []
                    S[(n,j)] = ''
                    for m in range(1, n-j):
                        t.append('[\\Omega_{'+str(n-m)+"}(t'), S_{"+str(n-m)+'}^{'+str(j)+"}](t')")
                    S[(n,j)] = '+'.join(t)
                    temp.append('\\frac{B_{'+str(i)+'}}{{'+str(i)+'}!}\\left('+'+'.join(t)+'\\right)')
            st[f'\\Omega_{k}(t)'] = f"\\int_{'{t_0}'}^{'{t}'}dt' \\left[{' + '.join(temp)}\\right]"
    for Omega, Value in st.items():
        if(disp_):
            disp(md(f"{Omega}={Value}"))
        elif(print_):
            print(f"{Omega}={Value}\n\n\n")
    return st, S
