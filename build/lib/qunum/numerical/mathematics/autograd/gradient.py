from torch.autograd import grad
from torch import jit, Tensor, swapaxes as SW
from ...physics.quantum.qobjs.torch_qobj import TQobj
from itertools import product
import torch
from typing import Tuple, List, Optional
def Dx(
        y:TQobj, 
        x:TQobj|Tensor,
        der_dim:int|None = None,
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True,
        symmetric: bool = False
    )->Tensor|TQobj:
    if not isinstance(y, TQobj):
        assert isinstance(y,TQobj), 'Not Implemented for just Torch Tensors yet only TQobjs'
        return None 
    elif not (y.is_complex()):
        if(y._metadata.obj_tp == 'operator' and symmetric):
            return (
                TQobj(
                    Dop(
                        y, 
                        x, 
                        der_dim,
                        retain_graph=retain_graph, 
                        create_graph=create_graph,
                        allow_unused=allow_unused, 
                    ), 
                    meta = y._metadata
                )
            )
        else:
            return (
                TQobj(
                    DGen(
                        y, 
                        x,
                        y._metadata.obj_tp, 
                        der_dim,
                        retain_graph=retain_graph, 
                        create_graph=create_graph,
                        allow_unused=allow_unused, 
                    ), 
                    meta = y._metadata
                )
            )
    else:
        return (
                TQobj(
                    DGenComplex(
                        y, 
                        x,
                        y._metadata.obj_tp, 
                        der_dim,
                        retain_graph=retain_graph, 
                        create_graph=create_graph,
                        allow_unused=allow_unused, 
                    ), 
                    meta = y._metadata
                )
            )

@torch.jit.script
def Dop(
        Op:TQobj,
        x:Tensor, 
        der_dim:int, 
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True 
    )->Tensor:
    Op = torch.swapaxes(torch.swapaxes(Op, axis0=0, axis1=-1), axis0=1, axis1=-2)
    G = torch.zeros_like(Op)
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones_like(Op[0,0]) ]
    for i in range(Op.shape[0]):
        for j in range(i, Op.shape[1]):
            t = grad(
                [ Op[i, 0] ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
            if(t is not None):
                G[i,j] += t[der_dim]
    return torch.swapaxes(torch.swapaxes(G, axis0=-2, axis1=1,), axis0=-1, axis1=0)

@torch.jit.script
def DGen(
        V:TQobj,
        x:Tensor,
        ket_or_bra:str,  
        der_dim:int, 
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True,
    )->Tensor:
    if(ket_or_bra == 'ket'):
        swap1:Tuple[int, int] = (0, -2)
        swap2:Tuple[int, int] = (1, -1)
    else:
        swap1:Tuple[int, int] = (0, -1)
        swap2:Tuple[int, int] = (1, -2)
    V = torch.swapaxes(torch.swapaxes(V, axis0 = swap1[0], axis1 = swap1[1] ), axis0 = swap2[0], axis1 = swap2[1])
    G = torch.zeros_like(V)
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones_like(V[0,0]) ]
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            t = grad(
                [ V[i, j] ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
            if(t is not None):
                G[i,j] = G[i,j] + t[der_dim]
    return torch.swapaxes(torch.swapaxes(G, axis0 = swap2[1], axis1 = swap2[0]), axis0 = swap1[1], axis1 = swap1[0])

@jit.script
def DGenComplex(
        V:TQobj,
        x:Tensor,
        ket_or_bra:str,  
        der_dim:int, 
        retain_graph: bool = True, 
        create_graph: bool = False, 
        allow_unused: bool = True,
    )->Tensor:
    if(ket_or_bra == 'ket'):
        swap1:Tuple[int, int] = (0, -2)
        swap2:Tuple[int, int] = (1, -1)
    else:
        swap1:Tuple[int, int] = (0, -1)
        swap2:Tuple[int, int] = (1, -2)
    V = SW(SW(V, axis0 = swap1[0], axis1 = swap1[1] ), axis0 = swap2[0], axis1 = swap2[1])
    G = torch.zeros_like(V)
    grad_outputs: List[Optional[torch.Tensor]] = [ torch.ones_like(V[0,0]).real ]
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            tr = grad(
                [ V[i, j].real ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
            ti = grad(
                [ V[i, j].imag ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=retain_graph, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
            if(tr is not None and ti is not None):
                if(len(x.shape) > 1):
                    G[i,j] = G[i,j] + torch.complex(tr[:,der_dim].real, ti[:,der_dim].real)
                else:
                    G[i,j] = G[i,j] + torch.complex(tr[der_dim].real, ti[der_dim].real)
    return SW(SW(G, axis0 = swap2[1], axis1=swap2[0]), axis0 = swap1[1], axis1 = swap1[0])