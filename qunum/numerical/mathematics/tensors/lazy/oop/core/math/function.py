from typing import Callable, Self, Any
from torch import Tensor
class LazySqrt:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            depth:int
        )->Self:
        self.A = A
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).sqrt()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .sqrt()) with exception {e}'.format(dep = str(self.depth), e = str(e)))
class LazyExp:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            depth:int
        )->Self:
        self.A = A
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).exp()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .exp()) with exception {e}'.format(dep = str(self.depth), e = str(e)))
class LazySin:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            depth:int
        )->Self:
        self.A = A
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).sin()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .sin()) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazyCos:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            depth:int
        )->Self:
        self.A = A
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).cos()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .cos()) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazyLog:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            depth:int
        )->Self:
        self.A = A
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).log()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .log()) with exception {e}'.format(dep = str(self.depth), e = str(e)))
        
