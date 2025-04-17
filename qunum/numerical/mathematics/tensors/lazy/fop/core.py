import torch
from torch import Tensor, tensor as createTensor, device as torchDevice, dtype as torchDtype, complex128 as complex128, Size, einsum, tensordot as contract
from typing import Self, Any, Callable, Iterable
from ....combintorix import EnumerateArgCombos


class LazyTensor:
    def __init__(
            self,
            ProjectorFunction:Callable[[tuple[Any|Tensor], dict[Tensor|Any]], Tensor], 
            shape:Iterable[int]|None = None, 
            default_infer:Tensor= createTensor([0.]), 
            infer:bool = False,
            depth:int = 1
        )->Self:
        self.ProjectorFunction:Callable[[Tensor],Tensor] = ProjectorFunction
        if(infer): self.shape:Size  = self.ProjectorFunction(default_infer).shape
        else: self.shape:Size|None = shape
        self.depth = depth
        return 
    
    def __call__(self, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Self:
        return self.ProjectorFunction(*args, **kwargs)
    
    def to(self, **kwargs:dict[str:Any])->None:
        def doTo(self:LazyTensor, depth:int, kwargs:dict,  *args:tuple[Tensor|Any], **kwars:dict[Tensor|Any])->Tensor:
            try:
                return self(*args, **kwars).to(**kwargs)
            except Exception as e:
                raise Exception('Error at depth {d}, Error MSG: {err}'.format(d = depth, err = e))
        return LazyTensor(lambda *args, **kwars: doTo(self,self.depth, kwargs, *args, **kwars), depth = self.depth+1)

    def conj(self):
        def doConj(self:LazyTensor, depth:int, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            try:
                return self(*args, **kwargs).conj()
            except Exception as e:
                raise Exception('Error at depth {d}, Error MSG: {err}'.format(d = depth, err = e))

        return LazyTensor(lambda *args, **kwargs: doConj(self,self.depth, *args, **kwargs), depth = self.depth+1)
 
    def __getitem__(self, idx:EnumerateArgCombos)->Self:
        def doIndexIn(self:LazyTensor, depth:int, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            try:
                match idx:
                    case idx if isinstance(idx, EnumerateArgCombos):
                        return self(*args, **kwargs)[idx.__tensor__()]
                    case _:
                        return  self(*args, **kwargs)[idx]
            except Exception as e:
                raise Exception('Error at depth {d}, Error MSG: {err}'.format(d = depth, err = e))
        return LazyTensor(lambda *args, **kwargs: doIndexIn(self,self.depth, idx, *args, **kwargs), depth = self.depth+1)
            
    def __mul__(self, B:Self)->Self:
        def doMul(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return  self(*args, **kwargs) * B
                case int():
                    return  self(*args, **kwargs) * B
                case float():
                    return self(*args, **kwargs) * B
                case complex():
                    return self(*args, **kwargs) * B
                case LazyTensor():
                    return self(*args, **kwargs) *  B(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doMul(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def __matmul__(self, B:Self|Tensor)->Self:
        def doMatMul(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex,
          *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return self(*args, **kwargs)@ B
                case LazyTensor():
                    return self(*args, **kwargs) @ B(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doMatMul(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
          
    def __rmatmul__(self, B:Self|Tensor)->Self:
        def dorMatMul(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex,
          *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return B @ self(*args, **kwargs)
                case LazyTensor():
                    return  B(*args, **kwargs) @ self(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                dorMatMul(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def __add__(self, B:Self|Tensor)->Self:
        def doAdd(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return self(*args, **kwargs) + B
                case int():
                    return self(*args, **kwargs) + B
                case float():
                    return self(*args, **kwargs) + B
                case complex():
                    return self(*args, **kwargs) + B
                case LazyTensor():
                    return self(*args, **kwargs) +  B(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doAdd(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def __radd__(self, B:Self|Tensor)->Self:
        return self.__add__(B)
    
    def __sub__(self, B:Self|Tensor)->Self:
        def doSub(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return self(*args, **kwargs) - B
                case int():
                    return self(*args, **kwargs) - B
                case float():
                    return self(*args, **kwargs) - B
                case complex():
                    return self(*args, **kwargs) - B
                case LazyTensor():
                    return self(*args, **kwargs) -  B(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doSub(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def __rsub__(self, B:Self|Tensor)->Self:
        def doRSub(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return B - self(*args, **kwargs) 
                case int():
                    return B - self(*args, **kwargs)
                case float():
                    return B - self(*args, **kwargs)
                case complex():
                    return B - self(*args, **kwargs)
                case LazyTensor():
                    return B(*args, **kwargs) - self(*args, **kwargs) 
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doRSub(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )

    def __rtruediv__(self, B:Self|Tensor)->Self:
        def doRTrueDiv(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return B/self(*args, **kwargs) 
                case int():
                    return B/self(*args, **kwargs)
                case float():
                    return B/self(*args, **kwargs)
                case complex():
                    return B/self(*args, **kwargs)
                case LazyTensor():
                    return B(*args, **kwargs)/self(*args, **kwargs) 
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doRTrueDiv(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
   
    def __truediv__(self, B:Self|Tensor)->Self:
        def doAdd(self:LazyTensor, B:LazyTensor|Tensor|int|float|complex, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            match B:
                case Tensor():
                    return self(*args, **kwargs) / B
                case int():
                    return self(*args, **kwargs) / B
                case float():
                    return self(*args, **kwargs) / B
                case complex():
                    return self(*args, **kwargs) / B
                case LazyTensor():
                    return self(*args, **kwargs) /  B(*args, **kwargs)
                case _:
                    raise TypeError('Must be Tensor, float, int, complex, or LazyTensor')
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doAdd(
                                    self,B, *args,**kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1)
            
    def sum(self, dim:int|None = None)->Self:
        def doSum(self:LazyTensor,*args:tuple[Tensor|Any], dim:int|None = None,  **kwargs:dict[Tensor|Any])->Tensor:
            return self(*args, **kwargs).sum(dim=dim)
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doSum(
                                    self, *args, dim= dim, **kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def prod(self, dim:int|None = None)->Self:
        def doProd(self:LazyTensor,*args:tuple[Tensor|Any], dim:int|None = None,  **kwargs:dict[Tensor|Any])->Tensor:
            return self(*args, **kwargs).prod(dim=dim)
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doProd(
                                    self, *args, dim= dim, **kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )

    def reshape(self, *args:Iterable[int]|Size)->Self:
        def doProd(self:LazyTensor, args:Iterable[int]|Size, *ars:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            return self(*ars, **kwargs).reshape(*args)
        return LazyTensor(
            (
                lambda *ars, **kwargs: 
                    errTrackCall(
                        (
                            lambda *ars, **kwargs:
                                doProd(
                                    self, args, *ars, **kwargs
                                    )
                        ), 
                        self.depth, *ars, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def sqrt(self)->Self:
        def doSqrt(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).sqrt()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doSqrt(
                                    self, *args, **kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def abssqr(self)->Self:
        def doAbsSqr(fx:LazyTensor, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
            a = fx(*args, **kwargs)
            return a*a.conj() 
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doAbsSqr(
                                    self, *args **kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def abs(self)->Self:
        def doAbs(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).abs()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doAbs(
                                    self, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def pow(self, B:Tensor|int|complex|float)->Self:
        assert isinstance(B,Tensor) or isinstance(B,complex) or isinstance(B,int) or isinstance(B,float), TypeError('Must be numerical value or Tensor')    
        def doPow(self:LazyTensor, B:Tensor|int|complex|float, *args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).pow(B)
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doPow(
                                    self, *args, **kwargs
                                    )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def __pow__(self, B:Tensor|int|complex|float)->Self:
        return self.pow(B)

    def exp(self)->Self:
        def doExp(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).exp()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doExp(
                                    self, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )

    def sin(self)->Self:
        def doSin(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).sin()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doSin(
                                    self, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def cos(self)->Self:
        def doCos(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).cos()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doCos(
                                    self, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def log(self)->Self:
        def doLog(self:LazyTensor,*args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            return  self(*args, **kwargs).log()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doLog(
                                    self, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )
    
    def logbase(self, base:int|float|Tensor)->Self:
        def doLog(self:LazyTensor, base:int|float|Tensor, *args:tuple[Tensor|Any],  **kwargs:dict[Tensor|Any])->Tensor:
            match base:
                case Tensor():
                    pass
                case float():
                    base = createTensor(base)
                case int():
                    base = createTensor(base)
                case complex():
                    base = createTensor(base)
                case _:
                    raise TypeError("Must be Tensor, float, int, or complex type, but got {tp}".format(tp=str(type(base))))
            return self(*args, **kwargs).log()/base.log()
        return LazyTensor(
            (
                lambda *args, **kwargs: 
                    errTrackCall(
                        (
                            lambda *args, **kwargs:
                                doLog(
                                    self, base, *args, **kwargs
                                )
                        ), 
                        self.depth, *args, **kwargs)
            ), 
            depth = self.depth+1 
        )

    def getattr(self, name:str, *args:tuple[Any], **kwargs:dict[Any])->Self:
        def getAttr(self:LazyTensor, name:str, in_args, in_kwargs, *args, **kwargs):
            return getattr(self(*args, **kwargs),name )(*in_args, **in_kwargs)
        return LazyTensor(
            (
                lambda *ars, **kwars: 
                    errTrackCall(
                        (
                            lambda *ars, **kwars:
                                getAttr(
                                    self, name, args, kwargs *ars, **kwars
                                )
                        ), 
                        self.depth, *ars, **kwars)
            ), 
            depth = self.depth+1 
        )
    
    def __repr__(self)->str:
        if(self.shape is None):
            s = 'Unknown'
        else:
            s = self.shape 
        return """LazyTensor(\\lambda (x^{{\\mu}}, y^{{\\mu}}=?).f^{{A}}(x^{{\\mu}}, y^{{\\mu}}=?), shape = {s}, depth = {d})""".format(s= s, d = str(self.depth))
    
   
def contractLazy(A:Tensor|LazyTensor,  B:Tensor|LazyTensor,  dims:int|Iterable[int] = 2, out:Tensor|None = None)->LazyTensor|Tensor:
    match (A,B):
        case (Tensor(), Tensor()):
            return contract(A,B, dims=dims, out=out)
        case (LazyTensor(),Tensor()):
            return LazyTensor(
                lambda *args,**kwargs:
                    errTrackCall(
                        lambda *args, **kwargs: 
                            contract(A(*args, **kwargs), B, dims=dims, out=out),
                        depth = A.depth,
                        *args, **kwargs
                    ), 
                    depth = A.depth+1 
                )
        case (Tensor(), LazyTensor()):
            return LazyTensor(
                lambda *args,**kwargs:
                    errTrackCall(
                        lambda *args, **kwargs: 
                            contract(A, B(*args, **kwargs), dims=dims, out=out),
                        depth = B.depth,
                        *args, **kwargs
                    ), 
                    depth = B.depth+1 
                )
        case (LazyTensor(), LazyTensor()):
            return LazyTensor(
                lambda *args,**kwargs:
                    errTrackCall(
                        lambda *args, **kwargs: 
                            contract(A(*args, **kwargs), B(*args, **kwargs), dims=dims, out=out),
                        depth = max([A.depth, B.depth]),
                        *args, **kwargs
                    ), 
                    depth = max([A.depth,B.depth])+1 
                )
        case _:
            raise TypeError('Must Be LazyTensor or torch.Tensor type')

def callIt(fun:Tensor|LazyTensor, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Tensor:
    match fun:
        case LazyTensor():
            return fun(*args,**kwargs)
        case Tensor():
            return fun
        case _:
            raise TypeError('Error must be Tensor or Lazy Tensor')

def einsumLazy(indicies:str, *args:Iterable[LazyTensor|Tensor])->LazyTensor|Tensor:
    D = max([a.depth if isinstance(a, LazyTensor) else 0 for a in args])
    return LazyTensor(
        (
            lambda *ars, **kwargs: 
                errTrackCall(
                    lambda *ars, **kwargs: 
                    einsum(
                        indicies,
                        *map(
                            lambda fun: 
                                callIt(fun, *ars,**kwargs), 
                            args
                        ) 
                    ),
                    depth = D, 
                    *ars, **kwargs
            )
        ), 
        depth = D+1
    )


def errTrackCall(fx, depth, *args, **kwargs):
    try:
        return fx(*args, **kwargs)
    except Exception as e:
        raise Exception('Error at depth {d}, Error MSG: {err}'.format(d = depth, err = e))

