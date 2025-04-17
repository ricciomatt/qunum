from ..obj import LazyTensor
from torch import Tensor, tensordot as contract, einsum
from typing import Iterable, Any, Self
from warnings import warn

def contractLazy(
        A:Tensor|LazyTensor,  
        B:Tensor|LazyTensor,  
        dims:int|Iterable[int] = 2, 
        out:Tensor|None = None,
        warnMe:bool = True
    )->LazyTensor|Tensor:
    assert (isinstance(A,LazyTensor) or isinstance(A, Tensor)) and (isinstance(B,LazyTensor) or isinstance(B, Tensor)), TypeError('Must be LazyTensor or Tensor args, but got combination: type(A)={a}, type(B)={b}'.format(a=str(type(A)), b=str(type(B))))
    assert (isinstance(out,Tensor) or out is None), TypeError('keyword arguement out must be Tensor or None, but got out={out}'.format(out=str(type(out))))
    if(isinstance(A,Tensor) and isinstance(B,Tensor)):
        if(warnMe): warn(Warning('Use regular tensordot from torch instead if args are both tensors'))
        return contract(A,B, dims=dims, out=out) 
    else:
        D = max(
                [
                    a.depth for a in [A,B] if isinstance(a, LazyTensor)
                ]
            )
        return LazyTensor(LazyContract(A, B, depth = D, dims=dims, out=out))

def callIt(fun:Tensor|LazyTensor, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
    match fun:
        case LazyTensor():
            return fun(*args,**kwargs)
        case Tensor():
            return fun
        case _:
            raise TypeError('Error must be Tensor or Lazy Tensor')

def einsumLazy(indicies:str, *args:Iterable[LazyTensor|Tensor], warnMe:bool = True)->LazyTensor|Tensor:
    D = [a.depth for a in args if isinstance(a, LazyTensor)]
    if(len(D)!= 0):
        D = max(D)
        return LazyTensor(LazyEinsum(indicies, D, *args), depth = D+1)    
    else:
        if(warnMe): warn(Warning('Use regular einsum from torch instead if args are all tensors'))
        return einsum(indicies, *args)

    
class LazyEinsum:
    def __init__(self, indicies:str, depth:int,  *tensors:tuple[Tensor|LazyTensor])->Self:
        self.indicies = indicies
        self.tensors = tensors
        self.depth = depth
        return 
    def __call__(self, *args:tuple[Any|Tensor], **kwargs:dict[Any|Tensor])->Tensor:
        try:
            return einsum(self.indicies, *map(lambda fun: callIt(fun, *args, **kwargs), self.tensors))
        except Exception as e:
            raise Exception("Error on lazyEinsum({ind}, {args}, depth={depth})".format(ind=str(self.indicies), args = str(self.tensors), depth = self.depth))

class LazyContract:
    def __init__(self, A:Tensor|LazyTensor, B:Tensor|LazyTensor, depth:int, dims:int|Iterable[int] = 2, out:Tensor|None = None)->Self:
        self.A = A
        self.B = B
        self.dims = dims
        self.out = out
        self.depth = depth
        return 
    def __call__(self, *args:tuple[Any|Tensor], **kwargs:dict[Any|Tensor])->Tensor:
        try:
            match (self.A, self.B): 
                case (LazyTensor(),Tensor()):
                    return contract(self.A(*args, **kwargs), self.B, dims = self.dims, out=self.out)
                case (Tensor(), LazyTensor()):
                    return contract(self.A, self.B(*args, **kwargs), dims = self.dims, out=self.out)
                case (LazyTensor(), LazyTensor()):
                    return contract(self.A(*args, **kwargs), self.B(*args, **kwargs), dims = self.dims, out=self.out)
                case _:
                    raise TypeError('Must Be LazyTensor or torch.Tensor type')
        except Exception as e:
            raise Exception("Error on lazyEinsum({ind}, {args},dims={dims}, out={out}, depth={depth})".format(ind=str(self.A), args = str(self.B), depth = self.depth, out=self.out,dims=self.dims))