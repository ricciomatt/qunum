from torch.autograd import grad
from torch import jit, Tensor, swapaxes as SW
from ..combintorix import LazyEnumIndex
from itertools import product
import torch
from typing import Tuple, List, Optional
from numpy import argmax
def DxTen(
        y:Tensor,
        x:Tensor,
        der_dim:int|None = None,
        retain_graph: bool = True,
        create_graph: bool = True,
        allow_unused: bool = True,
    )->Tensor:
    assert isinstance(y,Tensor), 'For TQobjs use Dx(y, x,....)'
    if not (y.is_complex()):
        return DxRealTen(y, x, der_dim, retain_graph, create_graph, allow_unused)
    else:
        return DxComplexTen(y, x, der_dim, retain_graph, create_graph, allow_unused)

#@jit.script
def DxRealTen(
    y:Tensor,
    x:Tensor,
    der_dim:int,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool
)->Tensor:
    swix = torch.argmax(torch.tensor(y.shape))
    y = SW(y, axis0=0, axis1=swix)
    dy_dx = torch.zeros_like(y)
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones(y.shape[0], dtype=y.dtype) ]
    for i in LazyEnumIndex(dy_dx[0]):
        t = grad(
                [ y[:, *i] ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        if(t is not None):
            dy_dx[:,*i] += t[der_dim]
    return SW(dy_dx, axis0=0, axis1=swix)

#@jit.script
def DxComplexTen(
    y:Tensor,
    x:Tensor,
    der_dim:int,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool
)->Tensor:
    swix = torch.argmax(torch.tensor(y.shape))
    y = SW(y, axis0=0, axis1=swix)
    dy_dx = torch.zeros_like(y)
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones(y.shape[0], dtype=y.dtype) ]
    for i in LazyEnumIndex(dy_dx[0]):
        tr = grad(
                [ y[:, *i].real ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        ti = grad(
                [ y[:, *i].imag], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        
        if(tr is not None and ti is not None):
            if(len(x.shape) > 1):
                dy_dx[:,*i] = dy_dx[:,*i] + torch.complex(tr[:,der_dim].real, ti[:,der_dim].real)
            else:
                dy_dx[:,*i] = dy_dx[:,*i] + torch.complex(tr.real, ti.real)
    return SW(dy_dx, axis0=0, axis1=swix)
