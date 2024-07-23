
from torch.autograd import grad
from torch import Tensor, jit, zeros, ones, complex as complex_, reshape
from typing import List, Optional
from ...physics.quantum.qobjs.dense.core.torch_qobj import TQobj
@jit.script
def DxCQobj(
        y:TQobj,
        x:Tensor,
        order:int,
        der_dim:Tensor, 
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True,
    )->Tensor:
    shp = list(y.shape)
    if(len(der_dim.shape) != 0):
        shp.append(len(der_dim.shape))
    l = 1
    for s in shp:
        l*=s
    y = y.flatten()
    dy_dx = zeros(l, dtype = y.dtype, device = y.device).flatten()
    grad_outputs: List[Optional[Tensor]] = [ 
        ones(y.shape, dtype=y.dtype, device=y.device).real
    ]
    for o in range(order):
        rt:bool = bool(retain_graph or o != order-1) 
        if(o != 0):
            dy_dx = zeros(l, dtype = y.dtype, device = y.device).flatten()
        tr = grad(
                [ y.real ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        ti = grad(
                [ y.imag ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        if(tr is not None and ti is not None):
            if(len(tr.shape) != 1):
                dy_dx = dy_dx + complex_(tr[:,der_dim], ti[:,der_dim]).flatten()
            else:
                dy_dx = dy_dx + complex_(tr, ti).flatten()
        if(o != order-1):
            y = dy_dx.clone()
    return (reshape(dy_dx, shp))


@jit.script
def DxRQobj(
        y:TQobj,
        x:Tensor,
        order:int,
        der_dim:Tensor, 
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True,
    )->Tensor:
    shp = list(y.shape)
    if(len(der_dim.shape) != 0):
        shp.append(len(der_dim.shape))
    l = 1
    for s in shp:
        l*=s
    y = y.flatten()
    dy_dx = zeros(l, dtype = y.dtype, device = y.device).flatten()
    grad_outputs: List[Optional[Tensor]] = [ 
        ones(y.shape, dtype=y.dtype, device=y.device).real
    ]
    for o in range(order):
        rt:bool = bool(retain_graph or o != order-1) 
        if(o != 0):
            dy_dx = zeros(l, dtype = y.dtype, device = y.device).flatten()
        t = grad(
                [ y ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        
        if(t is not None):
            if(len(t.shape) != 1):
                dy_dx = dy_dx + t[:,der_dim].flatten()
            else:
                dy_dx = dy_dx + t.flatten()
        if(o != order-1):
            y = dy_dx.clone()
    return (reshape(dy_dx, shp))

