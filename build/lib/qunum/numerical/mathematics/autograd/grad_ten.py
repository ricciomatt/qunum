from torch.autograd import grad
from torch import jit, Tensor, swapaxes as SW
from ..combintorix import LazyEnumIndex
import torch
from typing import List, Optional
from torch import Tensor, jit, reshape
def DxTen(
        y:Tensor,
        x:Tensor,
        order:int|None = 1,
        der_dim:int|None = 0,
        retain_graph: bool = True,
        create_graph: bool = True,
        allow_unused: bool = True,
    )->Tensor:
    assert isinstance(y,Tensor), 'For TQobjs use Dx(y, x,....)'
    if not (y.is_complex()):
        return DxRTen(y, x, der_dim, order, retain_graph, create_graph, allow_unused)
    else:
        return DxCTen(y, x, der_dim, order, retain_graph, create_graph, allow_unused)

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
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones(y.shape[0], dtype=y.dtype, device=y.device) ]

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

@jit.script
def DxRTen(
    y:Tensor,
    x:Tensor,
    order:int, 
    der_dim:int,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool
)->Tensor:
    shp = y.shape
    y = y.flatten()
    dy_dx = torch.zeros_like(y)
    grad_outputs: List[Optional[torch.Tensor]] = [ 
        torch.ones(y.shape[0], dtype=y.dtype, device=y.device) 
    ]
    for o in range(order):
        rt:bool = bool(retain_graph or o != order-1)
        dy_dx = torch.zeros_like(y)
        t = grad(
                [ y ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        if(t is not None):
            dy_dx = dy_dx + t[:,der_dim]
        if(o != order-1):
            y = dy_dx.clone()
    return reshape(dy_dx, shp)
@jit.script
def DxCTen(
    y:Tensor,
    x:Tensor,
    order:int, 
    der_dim:int,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool
)->Tensor:
    y = torch.view_as_real(y)
    shp = y.shape
    y = y.flatten()
    dy_dx = torch.zeros_like(y)
    grad_outputs: List[Optional[torch.Tensor]] = [ 
        torch.ones(y.shape[0], dtype=y.dtype, device=y.device) 
    ]
    for o in range(order):
        rt:bool = bool(retain_graph or o != order-1)
        dy_dx = torch.zeros_like(y)
        t = grad(
                [ y ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        if(t is not None):
            dy_dx = dy_dx + t[:,der_dim]
        if(o != order-1):
            y = dy_dx.clone()
    return torch.view_as_complex(reshape(dy_dx, shp))
