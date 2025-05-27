from typing import Callable,Any
from torch import Tensor, vmap, from_numpy
from functools import wraps
from numpy import ndarray



def wrapScipy(*outer_args, **outer_kwargs)->Callable[[tuple[ndarray|Any], dict[ndarray|Any]], Tensor]:
    def wrap_inner(fn:Callable[[tuple[ndarray|Any], dict[ndarray|Any]], ndarray])->Callable[[tuple[ndarray|Any], dict[ndarray|Any]], Tensor]:
        @wraps(fn)
        def scipyLinalgFunction(*args:tuple, **kwargs:dict)->Tensor:
            args = list(args)
            for i in range(len(args)):
                if(isinstance(args[i], Tensor)):
                    args[i] = args[i].detach().cpu().numpy()
            result = fn(*args, **kwargs)
            return tuple(from_numpy(r) for r in result)
        return scipyLinalgFunction
    return wrap_inner
from scipy.linalg import hessenberg, schur

@wrapScipy()
def torch_schur(*args, **kwargs):
    return schur(*args, **kwargs)

@wrapScipy()
def torch_hessenberg(*args, **kwargs):
    return hessenberg(*args,**kwargs)

