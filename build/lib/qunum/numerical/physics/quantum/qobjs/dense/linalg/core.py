from typing import Callable,Any
from torch import Tensor
from ..torch_qobj import TQobj
from functools import wraps

def retTQobj(fun:Callable[[Tensor, tuple, dict], Tensor])->Callable[[TQobj, tuple, dict], TQobj]:
    def doit(a:TQobj, *args, **kwargs)->TQobj:
        assert isinstance(a,TQobj), TypeError('Must be TQobj')
        S:Tensor = fun(a.to_tensor(), *args, **kwargs)
        return TQobj(S, meta = a._metadata)
    return doit

def retTQobjTensor(fun:Callable[[Tensor, tuple, dict], Tensor])->Callable[[TQobj, tuple, dict], tuple[TQobj, Tensor]]:
    def doit(a:TQobj, *args, **kwargs):
        A:tuple[Tensor, Tensor] = fun(a.to_tensor(), *args , **kwargs)
        return A[0], TQobj(A, meta = a._metadata)
    return doit

def retTensor(fun:Callable[[Tensor, tuple, dict], Tensor])->Callable[[TQobj, tuple, dict], Tensor]:
    def doit(a:TQobj, *args, **kwargs):
        A:Tensor = fun(a.to_tensor())
        return A
    return doit



def fofMatrix(*outer_args, save_eigen:bool = False, recompute:bool = False, **outer_kwargs)->Callable[[TQobj, tuple, dict], TQobj]:
    def wrap_inner(fn):
        @wraps(fn)
        def functionOfMatrixDoIt(A:TQobj, *args:tuple, **kwargs:dict)->TQobj:
            V, U = A.diagonalize(inplace=False, ret_unitary= True,save_eigen=save_eigen, recompute=recompute)
            return U @ fn(V, *args, **kwargs) @ U.dag()
        return functionOfMatrixDoIt
    return wrap_inner
