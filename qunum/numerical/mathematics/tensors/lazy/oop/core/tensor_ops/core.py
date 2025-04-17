from typing import Callable, Self, Any, Iterable
from ......combintorix import EnumerateArgCombos
from torch import Tensor, dtype as torchDtype
class LazyGetItem:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            Idx:Iterable[int]|EnumerateArgCombos|int,
            depth:int
        )->Self:
        self.A = A
        self.Idx = Idx
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            match (self.Idx):
                case EnumerateArgCombos():
                    return self.A(*args, **kwargs)[self.Idx.__tensor__()]
                case _:
                    return self.A(*args, **kwargs)[self.Idx]
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=IndexIn(A, idx=...) with exception {e}'.format(dep = self.depth, e = str(e)))
        
class LazyReshape:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            shape:Iterable[int]|EnumerateArgCombos|int,
            depth:int
        )->Self:
        self.A = A
        self.shape = shape
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return self.A(*args, **kwargs).reshape(self.shape)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=IndexIn(A, idx=...) with exception {e}'.format(dep = self.depth, e = str(e)))


class LazyGetAttr:
    def __init__(
            self, 
            A:Callable[[tuple[Tensor|Any],dict[Tensor|Any]], Tensor], 
            name:str,
            depth:int,
            *args:tuple[Any],
            **kwargs:dict[Any]
        )->Self:
        self.A = A
        self.args = args
        self.kwargs = kwargs
        self.depth = depth
        self.name = name
        return 
    
    def __call__(self, *args:tuple, **kwargs:dict)->Tensor:
        try:
            return getattr(self.A(*args, **kwargs), self.name)(*self.args, **self.kwargs)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.{name}(*args={args}, **kwargs={kwargs}) with exception {e}'.format(dep = self.depth, e = str(e), name=self.name, args = self.args, kwargs = self.kwargs))
        
class LazyTo:
    def __init__(
            self,
            A:Callable[[tuple[Tensor|Any], dict[Tensor, Any]], Tensor],
            depth:int,
            *args:tuple[Any],
            **kwargs:tuple[Any]
            )->Self:
        self.A:Callable[[tuple[Tensor|Any], dict[Tensor, Any]], Tensor] = A
        self.args = args
        self.kwargs = kwargs
        self.depth = depth 
        return 
    def __call__(self, *args:tuple[Any], **kwargs:dict[Any])->Tensor:
        try:
            return self.A(*args, **kwargs).to(*self.args, **self.kwargs)
        except Exception as e:
            raise Exception('Error computing LazyTensor at (depth {dep}, Oper=.to(*args={args}, **kwargs={kwargs}) with exception {e}'.format(dep = self.depth, e = str(e), args = self.args, kwargs = self.kwargs))
        