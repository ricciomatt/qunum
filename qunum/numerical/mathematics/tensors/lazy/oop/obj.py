import torch
from torch import Tensor, dtype as torchDtype, complex128 as complex128, Size, device as torchDevice
from typing import Self, Any, Callable, Iterable, Sequence
from ....combintorix import EnumerateArgCombos
from .core import *
from copy import copy

class LazyTensor:
    def __init__(
            self,
            init_function:Callable[[tuple[Tensor|Any], dict[Tensor|Any]], Tensor], 
            depth:int = 1,
            dtype:torchDtype = torch.complex128,
            device:torchDevice ='cpu'
        )->Self:
        self.ProjectorFunction:Callable[[tuple[Tensor|Any], dict[Tensor|Any]], Tensor] = init_function
        self.depth = depth
        self.shape = None
        self.dtype = dtype
        self.device = device
        return 
    
    def __call__(self, *args:tuple[Tensor|Any], **kwargs:dict[Tensor|Any])->Self:
        return self.ProjectorFunction(*args, **kwargs)
    
    def to(self, *args:tuple[Any],  inplace:bool = False, **kwargs:dict[Any])->None:
        if(inplace):
            self.ProjectorFunction = LazyTo(self.ProjectorFunction, copy(self.depth), *args, **kwargs)
            self.depth+=1
            return
        return LazyTensor( 
            LazyTo(self, self.depth, *args, **kwargs), 
            depth = self.depth+1
        )
              
    def __mul__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper A*B')
        return LazyTensor( 
            LazyMul(self, B, True, isinstance(B,LazyTensor), self.depth), 
            depth = self.depth+1
        )
    
    def __rmul__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper A*B')
        return LazyTensor( 
            LazyMul(B, self, isinstance(B,LazyTensor), True, self.depth), 
            depth = self.depth+1
        )
    
    def __matmul__(self, B:Self|Tensor)->Self:

        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor) )), TypeError('Must be LazyTensor or Tensor for oper A@B')
        return LazyTensor( 
            LazyMatMul(self, B, True, isinstance(B,LazyTensor), self.depth), 
            depth = self.depth+1
        )
          
    def __rmatmul__(self, B:Self|Tensor)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor) )), TypeError('Must be LazyTensor or Tensor for oper A@B')
        return  LazyTensor( 
            LazyMatMul(B, self, isinstance(B,LazyTensor), True, self.depth), 
            depth = self.depth+1
        )
    
    def __add__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper A+B')
        return  LazyTensor( 
            LazyAdd(self, B, True, isinstance(B,LazyTensor), self.depth), 
            depth = self.depth+1
        )
    
    def __radd__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper B+A')
        return  LazyTensor( 
            LazyAdd(B, self, isinstance(B,LazyTensor),True, self.depth), 
            depth = self.depth+1
        )
    
    def __sub__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper A-B')
        return  LazyTensor( 
            LazySub(self, B, True, isinstance(B,LazyTensor), self.depth), 
            depth = self.depth+1
        )
     
    def __rsub__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper B-A')
        return  LazyTensor( 
            LazySub(B, self, isinstance(B,LazyTensor),True, self.depth), 
            depth = self.depth+1
        )
    
    def __truediv__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper A/B')
        return  LazyTensor( 
            LazyDiv(self, B, True, isinstance(B,LazyTensor), self.depth), 
            depth = self.depth+1
        )
    
    def __rtruediv__(self, B:Self|Tensor|int|float|complex)->Self:
        assert any((isinstance(B, Tensor),isinstance(B, LazyTensor),isinstance(B, float),isinstance(B, int), isinstance(B, complex),)), TypeError('Must be LazyTensor, Tensor, or python numeric for oper B/A')
        return  LazyTensor( 
            LazyDiv(B, self, isinstance(B,LazyTensor),True, self.depth), 
            depth = self.depth+1
        )
             
    def sum(self, dim:Size|Iterable[int]|int|None = None, keepdim:bool= False, dtype:torchDtype|None=None)->Self:
        return LazyTensor(LazySum(self, dim = dim, depth = self.depth, keepdim = keepdim, dtype = dtype), depth=self.depth+1)
    
    def __getitem__(self, indicies:Iterable[int]|EnumerateArgCombos)->Self:
        return LazyTensor(LazyGetItem(self, Idx=indicies, depth=self.depth), depth=self.depth+1)
    
    def prod(self, dim:Size|Iterable[int]|int|None = None, keepdim:bool= False, dtype:torchDtype|None=None)->Self:
        return LazyTensor(LazyProd(self, dim = dim, depth = self.depth, keepdim = keepdim, dtype = dtype), depth=self.depth+1)
    
    def conj(self)->Self:
        return LazyTensor(LazyConj(self), depth=self.depth+1)

    def reshape(self, shape:Sequence[int]|Size)->Self:
       return LazyTensor(LazyReshape(self, shape, depth = self.depth), depth=self.depth+1)
    
    def sqrt(self)->Self:
        return  LazyTensor(LazySqrt(self, self.depth), depth=self.depth+1)
    
    def sin(self)->Self:
        return  LazyTensor(LazySin(self, self.depth), depth=self.depth+1)
    
    def cos(self)->Self:
        return  LazyTensor(LazyCos(self, self.depth), depth=self.depth+1)
    
    def exp(self)->Self:
        return  LazyTensor(LazyExp(self, self.depth), depth=self.depth+1)
    
    def log(self)->Self:
        return  LazyTensor(LazyLog(self, self.depth), depth=self.depth+1)
    
    def logbase(self, base)->Self:
        return  LazyTensor(LazyLogBase(self, Base =base, depth=self.depth), depth=self.depth+1)
    
    def getattr(self, name:str, *args:tuple[Any], **kwargs:dict[Any])->Self:
        assert name in dir(Tensor), KeyError('Method {name} does not exist, valid methods include {dir}'.format(name =str(name), dir = dir(Tensor)))  
        return  LazyTensor(LazyGetAttr(self,name, self.depth, *args, **kwargs), depth=self.depth+1)
    
    def abs(self)->Self:
        return  LazyTensor(LazyGetAttr(self, 'abs', self.depth), depth=self.depth+1)

    def abssqr(self)->Self:
        return  LazyTensor(LazyAbsSqr(self, depth=self.depth), depth=self.depth+1)
    
    def pow(self, exponenet:Tensor|int|complex|float)->Self:
        assert isinstance(exponenet,Tensor) or isinstance(exponenet,complex) or isinstance(exponenet,int) or isinstance(exponenet,float), TypeError('Must be numerical value or Tensor')    
        return LazyTensor(LazyPow(self, exponenet, depth=self.depth), depth=self.depth+1)
    
    def __pow__(self, B:Tensor|int|complex|float)->Self:
        return self.pow(B)

    def __repr__(self)->str:
        return """LazyTensor(\\lambda (x^{{\\mu}}, y^{{\\mu}}=?).f^{{A}}(x^{{\\mu}}, y^{{\\mu}}=?), depth = {d})""".format(d = str(self.depth))
    
    def display_attr_and_args(self)->str:
        tensor_class = torch.Tensor
        for method in dir(tensor_class):
            func = getattr(tensor_class, method)
            if callable(getattr(tensor_class, method)) and method not in dir(LazyTensor):
                doc = func.__doc__  # Fetch documentation
                print(f"Method: {method}")
                try:
                    # Print the first part of the docstring (usually includes arguments)
                    print(f"Docstring (summary): {doc.splitlines()[1]}\n")
                except:
                    pass
        return
    
    def diagonilize(self, get_U:bool = False, assume_hermitian:bool = False)->Self|tuple[Self,Self]:
        if get_U:
            return LazyTensor(LazyDiag(self, self.depth, assume_hermitian = assume_hermitian), self.depth+1), LazyTensor(LazyEig(self, self.depth,eigenvectors=True, justVectors=True, hermitian=assume_hermitian), self.depth+1)
        else:
            return LazyTensor(LazyDiag(self, self.depth, assume_hermitian= assume_hermitian), self.depth+1)
    
    def eig(self, eigenvectors:bool = False, assume_hermitian:bool = False, justVectors:bool = False, seperateLazy:bool = False)->Self|tuple[Self,Self]:
        if(seperateLazy):
            return LazyTensor(LazyEig(self, depth=self.depth, hermitian=assume_hermitian), self.depth+1), LazyTensor(LazyEig(self, depth=self.depth, eigenvectors=True, justVectors=True, hermitian=assume_hermitian), self.depth+1)
        else:
            return LazyTensor(LazyEig(self,self.depth, hermitian = assume_hermitian, eigenvectors=eigenvectors, justVectors=justVectors), self.depth+1)

    def applyFofM(self, fOfM:Callable[[Tensor],Tensor], assume_hermitian:bool = False )->Self:
        return LazyTensor(LazyApplyFofM(fOfM, depth=self.depth, hermitian=assume_hermitian), self.depth+1)