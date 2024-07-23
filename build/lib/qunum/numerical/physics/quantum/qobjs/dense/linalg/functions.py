from torch import  jit, Tensor, tensor, e
from .core import retTQobj, retTQobjTensor, retTensor, fofMatrix
from ..torch_qobj import TQobj
from torch.linalg import matrix_exp, matrix_norm, matrix_power, eig as teig, eigh as teigh, eigvals as teigvals, eigvalsh as teigvalsh, slogdet as tslogdet, det as tdet, cholesky as tcholesky, inv
from torch import tensordot, diag_embed


expm = retTQobj(matrix_exp)
mat_pow = retTQobj(matrix_power)
norm = retTQobj(matrix_norm)
eig = retTQobjTensor(teig)
eigh = retTQobjTensor(teigh)
eigvals = retTensor(teigvals)
eigvalsh = retTensor(teigvalsh)
slogdet = retTensor(tslogdet)
det = retTensor(tdet)
cholesky = retTQobj(tcholesky)
inv = retTQobj(inv)


def tensordot(A:TQobj, B:TQobj, dims:int = 2, out:None|Tensor = None)->TQobj:
    return TQobj(tensordot(A, B, dims=dims, out=out).to_tensor(), dims = A._metadata.dims)

def diagnolize_operator(A:TQobj,*args, ret_unitary:bool = True, **kwargs,)->tuple[TQobj, TQobj]|TQobj:
    if(ret_unitary):
        if(A._metadata.is_hermitian):
            v, U = eig(A)
        else:
            v, U = eigh(A)
        return TQobj(diag_embed(v), meta=A._metadata), U
    else:
        if(A._metadata.is_hermitian):
            v = eigvals(A)
        else:
            v = eigvalsh(A)
        return TQobj(diag_embed(v), meta = A._metadata)

@fofMatrix(save_eigen=False, recompute=False)
def sqrtm(A:TQobj, *args:tuple, **kwargs:dict)->TQobj:
    return A.sqrt()

@fofMatrix(save_eigen=False, recompute=False)
def logm(A:TQobj, **kwargs)->TQobj:
    return A.log()

@fofMatrix(save_eigen=True, recompute=False)
def logmbase(A:TQobj, *args, base:Tensor = tensor(e), **kwargs)->TQobj:
    return A.log()/base.log

@fofMatrix(save_eigen=False, recompute=False)
def inversem(A:TQobj, *args, **kwargs):
    return A.pow(-1)




