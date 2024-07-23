from typing import Any, Generator, Self,Iterable
from ..core.obj import PauliMatrix
import numpy as np 
import torch
class LazyJordanWigner:
    def __init__(self, NParticles:int, create_or_annhilate='annhilate', device:str|int='cpu', dtype:torch.dtype = torch.complex128 )->Self:
        self.N = NParticles
        self.create_or_annhilate = create_or_annhilate
        self.device = device
        self.dtype = dtype
        return
    
    def __call__(self, nix:Iterable[int]|int|None = None, generator:bool=True ) -> PauliMatrix|Generator[PauliMatrix, None, None]:
        if(generator):
            return self.__iter__(nix)
        else:
            return self.__pauli__(nix)
    
    def check_index(self, nix)->range|torch.Tensor|np.ndarray:
        if(nix is None):
            return range(0, self.N)
        elif(isinstance(nix,int)):
            if(nix<0):
                assert(self.N + nix < self.N and self.N+nix>0), IndexError(f'Index not in Range [0, {self.N}]')
                nix = self.N+nix
            else:
                assert(nix < self.N and nix>0), IndexError(f'Index not in Range [0, {self.N}]')
            return np.array([nix])
        if(isinstance(nix,slice) or isinstance(nix, range)):
            if(nix.step>0):
                assert (nix.start>0 and nix.stop<self.N), IndexError(f'Indicies not in Range [0, {self.N}]')
            else:        
                assert (nix.stop>0 and nix.start<self.N), IndexError(f'Indicies not in Range [0, {self.N}]')
            return nix
        elif(isinstance(nix, set)):
            return np.array(list(nix))
        elif(is_iter(nix)):
            if not (isinstance(nix,torch.Tensor)):
                nix = np.array(nix)
            assert nix.min()>0 and nix.max()<self.N, IndexError(f'Indicies not in Range [0, {self.N}]')
            return torch.from_numpy(nix)
        else:
            raise TypeError('Not valid Type provided')
        
    def __getitem__(self, nix:Iterable[int]|int)->PauliMatrix:
        return self(nix, generator = False)
    
    def __iter__(self , nix:None|int|Iterable[int])->Generator[PauliMatrix, None, None]:
        nix = self.check_index(nix)
        def yield_jw_trans(nix:Iterable[int], N:int, dtype:torch.dtype, create_or_annhilate:str, device:str|int)->Generator[PauliMatrix, None, None]:
            match create_or_annhilate:
                case 'annhilate':
                    v = -1j/2
                case 'create':
                    v = 1j/2
                case _:
                    raise ValueError("Keyword Arguement create_or_annhilate not valid must be in {'annhilate', 'create'} but got "+'"'+str(create_or_annhilate)+'"')
            for i in nix:
                basis = torch.zeros(1, N, 4, dtype = dtype, device = device)
                basis[0,:i, 0] = 1
                basis[0, i] = torch.tensor((0,1/2,v,0)).to(dtype= dtype, device= device)
                basis[0, :, 3] = 1
                yield PauliMatrix(basis, torch.tensor([complex(1, 0)], dtype= dtype, device=device))
        
        return yield_jw_trans(nix, self.N, dtype = self.dtype, device = self.device, create_or_annhilate=self.create_or_annhilate)

    def __pauli__(self, nix:int|Iterable[int]|None = None)->PauliMatrix:
        if(nix is None):
            return self.__getall__()
        return sum(
            (self.__iter__(nix))
        ) 
    
    def get_ix(self, nix:torch.Tensor|np.ndarray)->PauliMatrix:
        match self.create_or_annhilate:
            case 'annhilate':
                v = -1j/2
            case 'create':
                v = 1j/2
            case _:
                raise ValueError("Keyword Arguement create_or_annhilate not valid must be in {'annhilate', 'create'} but got "+'"'+str(self.create_or_annhilate)+'"')
        assert (isinstance(nix,torch.Tensor) or isinstance(nix,np.ndarray)), TypeError('Must numpy array or torch Tensor of ints')
        
        basis = torch.zeros((nix.shape[0], self.N, 4), dtype = self.dtype, device=self.device)
        basis[[torch.arange(nix.shape[0]), nix]] = torch.tensor((-1,0,0, 1), dtype= self.dtype, device= self.device)
        basis = basis.cumsum(dim=1)
        basis[[torch.arange(nix.shape[0]), nix]] = torch.tensor((-1,1/2,v,0),dtype = self.dtype, device = self.device)
        b = torch.zeros(4, dtype= self.dtype, device= self.device)
        b[0,0] = 0
        return PauliMatrix(basis = basis+b, coefs=torch.ones(basis.shape[0], dtype = self.dtype, device=self.device))

    def __getall__(self ):
        match self.create_or_annhilate:
            case 'annhilate':
                v = -1j/2
            case 'create':
                v = 1j/2
            case _:
                raise ValueError("Keyword Arguement create_or_annhilate not valid must be in {'annhilate', 'create'} but got "+'"'+str(self.create_or_annhilate)+'"')
        basis = torch.zeros((self.N, self.N, 4), dtype = self.dtype, device=self.device)
        basis[:,:,0] = 1.0+0j
        basis += (
            torch.einsum('AB,i->ABi', torch.eye(self.N, dtype = self.dtype, device=self.device), torch.tensor((-1,1/2,v, 0))) 
                +
            torch.einsum('AB,i->ABi', torch.tril(torch.ones(self.N,self.N,dtype = self.dtype, device=self.device)).fill_diagonal_(0.0 + 0.0j).T, torch.tensor((-1,0,0,1)))
        )
        return PauliMatrix(basis, torch.ones(basis.shape[0], dtype = self.dtype, device=self.device))
def is_iter(a):
    try:
        iter(a)
        return True
    except:
        return False