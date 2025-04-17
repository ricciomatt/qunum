from typing import Any, Self, Generator, Callable
from ......qobjs import TQobj, direct_prod
from ........mathematics.algebra.sun import get_pauli
from torch import stack, dtype as torchDtype, complex128, device as torchDevice

class JordanWignerSu2:
    def __init__(self, num_particles:int, assign_dims:dict[int:int]|None=None, dtype:torchDtype=complex128, device:torchDevice='cpu') -> Generator[TQobj, None, None]:
        self.num_particles = num_particles
        self.assign_dims = assign_dims
        self.dtype = dtype
        self.device = device
        return
    
    def __iter__(self, create_or_annihilate:str = 'annihilate')->Generator[TQobj,None,None]:
        assert create_or_annihilate in {'annihilate', 'create'}, ValueError("Key word arguement create_or_annhilate must be in the set {'create', 'annihilate'}")
        S:TQobj = get_pauli().to(dtype=self.dtype,device = self.device)
        match create_or_annihilate:
            case 'annihilate':
                return map(
                        lambda i: jrdn((S[1] - 1j*S[2])/2, S[0], S[3], i, self.num_particles)(), 
                        range(self.num_particles)
                    )
            case 'create':
                return map(
                        lambda i: jrdn((S[1] + 1j*S[2])/2, S[0], S[3], i, self.num_particles)(), 
                        range(self.num_particles)
                    )

    def __list__(self)->list[TQobj]:
        return list(self.__iter__())
    
    def __tuple__(self)->list[TQobj]:
        return tuple(self.__iter__())

    def __call__(self, create_or_annihilate:str = 'annihilate') -> TQobj:
        return TQobj(stack(list(self.__iter__(create_or_annihilate = create_or_annihilate))), dims = self.assign_dims)
    
    def __tqobj__(self) -> TQobj:
        return self()
    
    def __getitem__(self, ix:int)->TQobj:
        if(isinstance(ix, int)):
            assert ix < self.num_particles and ix >= 0, IndexError('Must be 0<=ix<{num_particles}'.format(num_particles = str(self.num_particles)))
            S:TQobj = get_pauli().to(dtype=self.dtype,device = self.device)
            return jrdn((S[1] - 1j*S[2])/2, S[0], S[3], ix, self.num_particles)()
        else:
            try:
                iter(ix)
            except:
                raise TypeError('Must be int or list of int')
            return TQobj(stack(list(map(lambda i: self[i], ix))), dims = self.assign_dims)
    



def jordan_wigner_su2(num_particles:int, assign_dims:dict[int:int]|None=None)->TQobj:
    S = get_pauli()
    return (
        TQobj(
           stack(
                list(map(
                    lambda i : jrdn((S[1] - 1j*S[2])/2, S[0], S[3], i, num_particles, )(), 
                    range(num_particles)
                ))
            ), 
        dims = assign_dims)
    )


def jrdn(op:TQobj,I:TQobj,s3:TQobj,i:int, n:int,)->Callable[[None], TQobj]:
    def fun(j:int, i:int, op:TQobj, I:TQobj, s3:TQobj):
        if(j>i):
            return I 
        elif(j<i):
            return s3
        else:
            return op
    A = lambda : direct_prod(*map(lambda x: fun(x, i, op, I, s3), range(n)))
    return A
       

