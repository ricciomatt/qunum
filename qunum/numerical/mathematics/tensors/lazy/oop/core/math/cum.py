from typing import Callable, Self, Any, Iterable
from torch import Tensor, dtype as torchDtype

class LazySum:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            dim:Iterable[int]|int,
            depth:int,
            keepdim:bool = False,
            dtype:torchDtype = None,
        )->Self:
        self.A = A
        self.dim = dim
        self.depth = depth
        self.keepdim = keepdim
        self.dtype = dtype
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).sum(dim=self.dim, keepdim=self.keepdim, dtype= self.dtype)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.sum(dim={dim}, keepdim={keepdim}, dtype={dtype}) with exception {e}'.format(dep = str(self.depth), e = str(e), keepdim=str(self.keepdim), dim=str(self.dim), dtype=str(self.dtype),))

class LazyProd:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            dim:Iterable[int]|int,
            depth:int,
            keepdim:bool = False,
            dtype:torchDtype = None,
        )->Self:
        self.A = A
        self.dim = dim
        self.depth = depth
        self.keepdim = keepdim
        self.dtype = dtype
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).prod(dim=self.dim, keepdim=self.keepdim, dtype= self.dtype)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.prod(dim={dim}, keepdim={keepdim}, dtype={dtype}) with exception {e}'.format(dep = str(self.depth), e = str(e), keepdim=str(self.keepdim), dim=str(self.dim), dtype=str(self.dtype),))

class LazyCumSum:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            dim:Iterable[int]|int,
            depth:int,
            dtype:torchDtype = None,
        )->Self:
        self.A = A
        self.dim = dim
        self.depth = depth
        self.dtype = dtype
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).cumsum(dim= self.dim, dtype= self.dtype)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.cumsum(dim={dim}, dtype={dtype}) with exception {e}'.format(dep = str(self.depth), e = str(e), dim=str(self.dim), dtype=str(self.dtype),))

class LazyCumProd:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            dim:Iterable[int]|int,
            depth:int,
            dtype:torchDtype = None,
        )->Self:
        self.A = A
        self.dim = dim
        self.depth = depth
        self.dtype = dtype
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).cumprod(dim= self.dim, dtype= self.dtype)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.cumprod(dim={dim}, dtype={dtype}) with exception {e}'.format(dep = str(self.depth), e = str(e), dim=str(self.dim), dtype=str(self.dtype),))