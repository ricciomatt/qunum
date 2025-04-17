from typing import Callable, Self, Any
from torch import Tensor, tensor as createTensor
class LazyLogBase:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            Base:int|float|complex|Tensor, 
            depth:int
        )->Self:
        self.A = A
        match Base:
            case Tensor():
                self.Base = Base
            case _:
                self.Base = createTensor(Base)
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args,**kwargs).log()/self.Base.log()
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .logbase(base={base})) with exception {e}'.format(dep = str(self.depth), base=self.Base, e = str(e)))

class LazyAbsSqr:
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
            m = self.A(*args,**kwargs)
            return m.conj()*m
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth ={dep}, Oper = .abssqr()) with exception {e}'.format(dep = str(self.depth), base=self.Base, e = str(e)))
