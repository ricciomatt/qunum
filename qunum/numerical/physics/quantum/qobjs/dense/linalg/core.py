from typing import Callable,Any
from torch import Tensor, vmap
from ..core.torch_qobj import TQobj
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



def fofMatrix(*outer_args, vmaped:bool = False, f_of_lambda:bool = True, save_eigen:bool = False, recompute:bool = False, check_hermitian:bool = False, use_inv:bool = False, **outer_kwargs)->Callable[[TQobj|Tensor, tuple, dict], TQobj]:
    def wrap_inner(fn:Callable[[TQobj|Tensor, tuple, dict], TQobj|Tensor])->Callable[[TQobj|Tensor , tuple, dict], TQobj]:
        @wraps(fn)
        def functionOfMatrixDoIt(A:TQobj, *args:tuple, **kwargs:dict)->TQobj:
            assert isinstance(A, TQobj), TypeError('Must be TQobj to compute fOfMatrix')
            if not (f_of_lambda):
                V, U = A.diagonalize(inplace=False, ret_unitary= True,save_eigen=save_eigen, recompute=recompute, check_hermitian=check_hermitian)
                V = fn(V, *args, **kwargs).diagnol(dim1=-1,dim2=-2).diag_embed()
                if use_inv:
                    return U.dag() @ V  @ U
                else:
                    return U.inv() @ V @ U
            else:
                V,U = A.eig(eigenvectors=True, save = save_eigen, recompute=recompute, check_hermitian=check_hermitian)
                if(vmaped):
                    def unPackDoit(v,U):
                        if(len(v.shape) == 1):
                            return vmap(lambda u: u*v)(U)
                        else:
                            return vmap(unPackDoit)(v,U)
                    if(use_inv):
                        return vmap(unPackDoit)(fn(V, *args, **kwargs), U.dag())@ U
                    else:
                        return vmap(unPackDoit)(fn(V, *args, **kwargs), U.inv())@ U
                else:
                    return U.inv() @ TQobj(fn(V, *args,**kwargs).diag_embed(), meta=U._metadata) @ U
        return functionOfMatrixDoIt
    return wrap_inner
