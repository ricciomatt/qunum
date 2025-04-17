from typing import Callable, Self, Any
from torch import Tensor
class LazyAdd:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor]|Tensor, 
            A_Call:bool,
            B_Call:bool,
            depth:int
        )->Self:
        self.A = A
        self.B = B
        self.A_Call = A_Call
        self.B_Call = B_Call
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.A_Call, self.B_Call):
                case (True, True):
                    return self.A(*args, **kwargs)+self.B(*args, **kwargs)
                case (True, False):
                    return self.A(*args, **kwargs)+self.B
                case (False, True):
                    return self.A + self.B(*args, **kwargs)
                case (False, False):
                    return self.A+self.B
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = +) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazySub:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor]|Tensor, 
            A_Call:bool,
            B_Call:bool,
            depth:int
        )->Self:
       
        self.A = A
        self.B = B
        self.A_Call = A_Call
        self.B_Call = B_Call
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.A_Call, self.B_Call):
                case (True, True):
                    return self.A(*args, **kwargs)-self.B(*args, **kwargs)
                case (True, False):
                    return self.A(*args, **kwargs)-self.B
                case (False, True):
                    return self.A-self.B(*args, **kwargs)
                case (False, False):
                    return self.A-self.B
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = -) with exception {e}'.format(dep = str(self.depth), e = str(e)))
        
class LazyMul:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor]|Tensor, 
            A_Call:bool,
            B_Call:bool,
            depth:int
        )->Self:
        self.A = A
        self.B = B
        self.A_Call = A_Call
        self.B_Call = B_Call
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.A_Call, self.B_Call):
                case (True, True):
                    return self.A(*args, **kwargs)*self.B(*args, **kwargs)
                case (True, False):
                    return self.A(*args, **kwargs)*self.B
                case (False, True):
                    return self.A*self.B(*args, **kwargs)
                case (False, False):
                    return self.A*self.B
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = *) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazyMatMul:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor]|Tensor, 
            A_Call:bool,
            B_Call:bool,
            depth:int
        )->Self:
        self.A = A
        self.B = B
        self.A_Call = A_Call
        self.B_Call = B_Call
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.A_Call, self.B_Call):
                case (True, True):
                    return self.A(*args, **kwargs)@self.B(*args, **kwargs)
                case (True, False):
                    return self.A(*args, **kwargs)@self.B
                case (False, True):
                    return self.A@self.B(*args, **kwargs)
                case (False, False):
                    return self.A@self.B
        except Exception as e:
           
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = @) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazyDiv:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor]|Tensor, 
            A_Call:bool,
            B_Call:bool,
            depth:int
        )->Self:
        self.A = A
        self.B = B
        self.A_Call = A_Call
        self.B_Call = B_Call
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.A_Call, self.B_Call):
                case (True, True):
                    return self.A(*args, **kwargs)/self.B(*args, **kwargs)
                case (True, False):
                    return self.A(*args, **kwargs)/self.B
                case (False, True):
                    return self.A / self.B(*args, **kwargs)
                case (False, False):
                    return self.A/self.B
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = /) with exception {e}'.format(dep = str(self.depth), e = str(e)))
class LazyPow:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            B:Tensor|float|int|complex, 
            depth:int
        )->Self:
        self.A = A
        self.B = B
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).pow(self.B)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = /) with exception {e}'.format(dep = str(self.depth), e = str(e)))

class LazyConj:
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
            return self.A(*args,**kwargs).conj()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .conj()) with exception {e}'.format(dep = str(self.depth), e = str(e)))