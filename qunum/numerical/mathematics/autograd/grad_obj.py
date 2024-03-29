from torch.autograd import grad
from torch import jit, Tensor, zeros, ones, complex as complex_, tensor, is_nonzero
from typing import List, Optional, Iterable, Any, Callable
from torch import Tensor, jit, reshape
from ...physics.quantum.qobjs.torch_qobj import TQobj
from ...seml.data.data_loaders.lazy.sampler import LazySampler
def unity(x)->Tensor|TQobj:
    return x

@jit.script
def DxRTen(
        y:Tensor,
        x:Tensor,
        order:int, 
        der_dim:Tensor,
        retain_graph: bool,
        create_graph: bool,
        allow_unused: bool
    )->Tensor:
    shp = list(y.shape)
    if(len(der_dim.shape) != 0):
        shp.append(len(der_dim.shape))
    
    l = 1
    for s in shp:
        l*=s
    y = y.flatten()
    grad_outputs: List[Optional[Tensor]] = [ 
        ones(y.shape[0], dtype=y.dtype, device=y.device) 
    ]
    dy_dx = zeros(l, dtype = y.dtype, device = y.device).flatten()
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
                dy_dx = dy_dx + (t[:,der_dim].flatten())
            else:
                dy_dx = dy_dx + t
        else:
            print(None)
        if(o != order-1):
            y = dy_dx.clone()
    return reshape(dy_dx, shp)


@jit.script
def DxCTen(
    y:Tensor,
    x:Tensor,
    order:int, 
    der_dim:Tensor,
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool
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
        tr = grad([ y.real ], 
                [ x ], 
                grad_outputs=grad_outputs, 
                retain_graph=rt, 
                create_graph=create_graph,
                allow_unused=allow_unused,
            )[0]
        ti = grad([ y.imag ], 
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



class D_Op:
    def __init__(
        self, 
        x:LazySampler|Tensor|None = None, 
        order:int = 1,
        der_dims:None|int|tuple[int,...]|Iterable[int]=0,
        retain_graph: bool = True,
        create_graph: bool = True,
        allow_unused: bool = True,
        fun_:Callable = unity,
        y_fun:Callable|None = None,
        n_iter:int = int(1e3),
        n_pts:int = int(1e2)
    )->None:
        assert x is None or isinstance(Tensor), TypeError('Must be None or Tensor type')
        self.x = x
        self.order = order
        if(der_dims is None):
            self.der_dims = tensor([0])[0]
        elif(isinstance(der_dims,int)):
            self.der_dims = tensor([der_dims])[0]
        elif(isinstance(der_dims, Tensor)):
            self.der_dims = der_dims
        else:
            self.der_dims = tensor(der_dims)
        self.kwargs = dict(retain_graph = retain_graph, create_graph = create_graph, allow_unused = allow_unused)
        self.fun_ = fun_
        self.y_fun = y_fun


        self.n_iter = n_iter
        self.n_pts = n_pts
        self.n = 0
        return
    
    def __call__(self, 
                 y:Tensor|TQobj, 
                 x:Tensor|None = None, 
                 der_dims:int|tuple[int,...] = None,
                 order:int|None = None,
                 **kwargs
                )->Tensor|TQobj:
        assert self.x is not None or x is not None, TypeError('Must have a tensor to take the derivative with respect to')
        assert isinstance(y, Tensor) or isinstance(y, TQobj), TypeError('Must be a Tensor or TQobj')
        assert isinstance(order, int) or isinstance(self.order, int), TypeError('Must be int')
        for k in self.kwargs:
            if(k in kwargs):
                self.kwargs[k] = kwargs[k]
            
        if(x is None):
            x = self.x
        assert isinstance(x,Tensor) and x.requires_grad, TypeError('X must be a tensor have a gradient')

        if not isinstance(order, int):
            order = self.order
        
        #Derivative Dims
        if(der_dims is None):
            assert self.der_dims is not None, ValueError('Derivative with respect to dimensions must be specified')
            der_dims = tensor(self.der_dims)
        elif(isinstance(der_dims, Tensor)):
            pass
        elif(isinstance(der_dims,int)):
            der_dims = tensor([der_dims])[0]
        else:
            der_dims = tensor(der_dims)
        
        #Tensor Derivatives
        if(isinstance(y, Tensor)):
            if(y.is_complex()):
                if(x.is_complex()):
                    assert not is_nonzero(x.imag).all() or not is_nonzero(x.real).all(), ValueError('Need to be either all real or all imaginary')
                A = (DxCTen(y, x, order, der_dims, **self.kwargs))
            else:
                A = (DxRTen(y, x, order, der_dims, **self.kwargs))
        #TQobj Derivatives
        elif(isinstance(y, TQobj)):
            y = TQobj(y)
            if(y.is_complex()):
                if(x.is_complex()):
                    assert not is_nonzero(x.imag).all() or not is_nonzero(x.real).all(), ValueError('Need to be either all real or all imaginary')
                A = (DxCQobj(y, x, order, der_dims, **self.kwargs))
            else:
                A = (DxRQobj(y, x, order, der_dims, **self.kwargs))
        return self.fun_(A)
    
    def lazy_der(self, y_fun:None|Callable = None, 
                 n:int|None = None,  
                 der_dims:int|tuple[int,...] = None,
                 order:int|None = None,
                 **kwargs)->Tensor:
        assert isinstance(self.x, LazySampler), 'To Have a lazy diff opertor must define LazyXsampler'
        assert self.y_fun is None or y_fun is None, 'To Have a lazy diff opertor must define a function to differentiate by passing y_fun or other'
        if(y_fun is None): 
            y_fun = self.y_fun
        x = self.get_x(n=n)
        y = y_fun(x)
        return self(y,x,der_dims=der_dims,order=order,**kwargs)

    def get_x(self, n:int|None= None)->Tensor:
        if(isinstance(self.x, LazySampler)):
            if(n is None):
                n = self.n_pts
            return self.x.sampleit(n)
        else:
            return self.x
    
    def __iter__(self)->object:
        return self
    
    def reset_iter(self)->None:
        self.n = 0
        return

    def __next__(self)->Tensor:
        if(self.n<self.n_iter):
            self.n+=1
            return self.lazy_der()
        else:
            self.reset_iter()
            raise StopIteration

    def __getitem__(self, n:int)->Tensor:
        A = next(self)
        self.reset_iter()
        return A
    
    def __repr__(self) -> str:
        return f"""D_Op(x = {self.x}, retain_graph = {self.kwargs['retain_graph']}, create_graph = {self.kwargs['create_graph']}, allow_unused = {self.kwargs['allow_unused']}, der_dims = {self.der_dims}, order = {self.order}, fun_ = {str(self.fun_)})"""
    